//! Assembling and patching, for the parts of Binary Ninja that write bytes back

use crate::insn;

const NOP: u8 = 0x01;
const DROP: u8 = 0x1a;

/// Indices have to be numeric: a symbolic name refers to a declaration this layer cannot see
pub fn assemble(text: &str) -> Result<Vec<u8>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }

    // The wrapper contributes the locals count and trailing `end` stripped off below
    let module =
        wat::parse_str(format!("(module (func {text}))")).map_err(|error| error.to_string())?;

    let body = function_body(&module).ok_or("assembled module has no function body")?;
    let (locals, rest) = body
        .split_first()
        .ok_or("assembled function body is empty")?;
    if *locals != 0 {
        return Err("locals cannot be declared here".to_owned());
    }

    let code = rest
        .strip_suffix(&[0x0b])
        .ok_or("assembled function body does not end where it should")?;

    // `wat` lets a block run to the end of the enclosing function, so `block` on its own comes
    // back missing its `end`, which patched in would restructure everything after it
    if !is_balanced(code) {
        return Err("instruction sequence opens or closes a block it does not finish".to_owned());
    }

    Ok(code.to_vec())
}

fn function_body(module: &[u8]) -> Option<&[u8]> {
    use wasmparser::{Parser, Payload};

    for payload in Parser::new(0).parse_all(module) {
        if let Ok(Payload::CodeSectionEntry(body)) = payload {
            let range = body.range();
            return module.get(range.start..range.end);
        }
    }

    None
}

fn is_balanced(mut code: &[u8]) -> bool {
    use wasmparser::Operator;

    let mut depth = 0i32;
    while !code.is_empty() {
        let Some(insn) = insn::decode(code) else {
            return false;
        };
        code = &code[insn.len..];

        match insn.op {
            Operator::Block { .. }
            | Operator::Loop { .. }
            | Operator::If { .. }
            | Operator::Try { .. }
            | Operator::TryTable { .. } => depth += 1,
            Operator::End | Operator::Delegate { .. } => depth -= 1,
            // A handler belongs to a block that has to already be open
            Operator::Else | Operator::Catch { .. } | Operator::CatchAll if depth == 0 => {
                return false;
            }
            _ => {}
        }

        if depth < 0 {
            return false;
        }
    }

    depth == 0
}

pub fn can_nop_out(data: &[u8]) -> bool {
    nop_replacement(data).is_some()
}

/// An instruction that consumed operands has to keep consuming them, so the replacement is one
/// `drop` per operand padded with `nop`
pub fn nop_out(data: &mut [u8]) -> bool {
    let Some((drops, len)) = nop_replacement(data) else {
        return false;
    };

    data[..len].fill(NOP);
    data[..drops].fill(DROP);
    true
}

fn nop_replacement(data: &[u8]) -> Option<(usize, usize)> {
    let insn = insn::decode(data)?;
    let arity = insn.arity()?;

    // A produced value has nothing to come from, and dropping a branch changes which code runs
    if arity.pushes != 0 || !insn.flow().falls_through() {
        return None;
    }

    let drops = arity.pops as usize;
    (drops <= insn.len).then_some((drops, insn.len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_single_instructions() {
        assert_eq!(assemble("nop").unwrap(), [0x01]);
        assert_eq!(assemble("i32.const 1").unwrap(), [0x41, 0x01]);
        assert_eq!(assemble("i32.const -1").unwrap(), [0x41, 0x7f]);
        assert_eq!(assemble("local.get 3").unwrap(), [0x20, 0x03]);
        assert_eq!(assemble("i32.add").unwrap(), [0x6a]);
        assert_eq!(
            assemble("i32.load offset=16 align=4").unwrap(),
            [0x28, 0x02, 0x10]
        );
    }

    #[test]
    fn assembles_sequences() {
        assert_eq!(
            assemble("i32.const 1 i32.const 2 i32.add").unwrap(),
            [0x41, 0x01, 0x41, 0x02, 0x6a]
        );
    }

    #[test]
    fn empty_input_assembles_to_nothing() {
        assert!(assemble("").unwrap().is_empty());
        assert!(assemble("   \n").unwrap().is_empty());
    }

    #[test]
    fn rejects_nonsense() {
        assert!(assemble("i32.frobnicate").is_err());
        assert!(assemble("i32.const").is_err());
        assert!(assemble("(module)").is_err());
    }

    #[test]
    fn rejects_an_unbalanced_sequence() {
        assert!(assemble("end").is_err());
        assert!(assemble("end nop").is_err());
        assert!(assemble("block").is_err());
    }

    #[test]
    fn rejects_a_handler_without_its_block() {
        assert!(assemble("else").is_err());
        assert!(assemble("nop else nop").is_err());
        assert!(assemble("if else end").is_ok());
    }

    #[test]
    fn assembles_balanced_blocks() {
        assert_eq!(assemble("block end").unwrap(), [0x02, 0x40, 0x0b]);
        assert_eq!(
            assemble("block (result i32) i32.const 0 end").unwrap(),
            [0x02, 0x7f, 0x41, 0x00, 0x0b]
        );
    }

    #[test]
    fn assembly_round_trips_through_the_decoder() {
        for text in [
            "nop",
            "i32.const 1234567",
            "i64.const -1",
            "f64.const 0x1p+0",
            "local.tee 0",
            "i32.load8_s offset=1",
            "memory.size",
            "v128.const i32x4 1 2 3 4",
            "i8x16.shuffle 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
        ] {
            let bytes = assemble(text).unwrap_or_else(|error| panic!("{text}: {error}"));
            let insn = insn::decode(&bytes).unwrap_or_else(|| panic!("{text} did not decode"));
            assert_eq!(insn.len, bytes.len(), "{text}");
        }
    }

    #[test]
    fn nopping_out_keeps_the_stack_balanced() {
        let mut store = assemble("i32.store").unwrap();
        assert_eq!(store.len(), 3);
        assert!(nop_out(&mut store));
        assert_eq!(store, [DROP, DROP, NOP]);

        let mut drop_op = assemble("drop").unwrap();
        assert!(nop_out(&mut drop_op));
        assert_eq!(drop_op, [DROP]);

        let mut nop = assemble("nop").unwrap();
        assert!(nop_out(&mut nop));
        assert_eq!(nop, [NOP]);
    }

    #[test]
    fn a_nopped_instruction_still_decodes_with_the_same_stack_effect() {
        for text in ["i32.store", "drop", "global.set 0", "i32.store8", "nop"] {
            let mut bytes = assemble(text).unwrap();
            let before = insn::decode(&bytes).unwrap().arity().unwrap();
            assert!(nop_out(&mut bytes), "{text} should be nop-able");

            let mut pops = 0;
            let mut rest = &bytes[..];
            while !rest.is_empty() {
                let insn = insn::decode(rest).unwrap_or_else(|| panic!("{text} left garbage"));
                pops += insn.arity().unwrap().pops;
                assert_eq!(insn.arity().unwrap().pushes, 0);
                rest = &rest[insn.len..];
            }
            assert_eq!(pops, before.pops, "{text} changed the stack");
        }
    }

    #[test]
    fn refuses_to_nop_what_it_cannot_replace() {
        for text in ["i32.const 1", "i32.add", "local.get 0", "return", "br 0"] {
            let mut bytes = assemble(text).unwrap();
            assert!(!can_nop_out(&bytes), "{text}");
            assert!(!nop_out(&mut bytes), "{text}");
        }
    }

    #[test]
    fn refuses_when_the_drops_would_not_fit() {
        let copy = assemble("memory.copy").unwrap();
        assert_eq!(copy.len(), 4);
        assert!(can_nop_out(&copy));

        assert!(!can_nop_out(&[0x36, 0x02, 0x10][..1]));
    }
}
