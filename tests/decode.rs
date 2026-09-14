//! Whole-decoder behaviour, exercised the way Binary Ninja exercises it: arbitrary bytes with no
//! promise that any of it is a real instruction

use binja_wasm::insn::{Instruction, MAX_INSTR_LEN, decode, decode_any};
use binja_wasm::{asm, cfg, lift};

/// Every input has to come back as a decode or a clean `None`, and the core goes on to ask for the
/// text, branches and semantics of whatever came back, so every accessor it can reach is called
#[test]
fn arbitrary_bytes_never_panic() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut buffer = [0u8; MAX_INSTR_LEN];

    for _ in 0..200_000 {
        for byte in &mut buffer {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }

        if let Some(insn) = decode(&buffer) {
            assert!((1..=buffer.len()).contains(&insn.len));
            exercise(&insn);
        }

        // What recovery decodes with, which has no length cap and so is reached by bytes the
        // capped decoder turns away
        if let Some(insn) = decode_any(&buffer) {
            assert!((1..=buffer.len()).contains(&insn.len));
            exercise(&insn);
        }

        // Recovery and patching read a run of bytes rather than one instruction
        let _ = cfg::recover(&buffer, 0x1000, None).blocks();
        let _ = asm::can_nop_out(&buffer);
        let _ = asm::nop_out(&mut buffer.clone());
    }
}

/// Everything the core asks about an instruction, all of which it must survive
fn exercise(insn: &Instruction) {
    assert!(!insn.mnemonic().is_empty());
    let _ = insn.operands();
    let _ = insn.arity();
    let _ = insn.flow();
    let _ = insn.operator_id();
    let _ = lift::stack_effect(insn, None);
}

/// A short read at the end of a section must not be mistaken for a shorter instruction that fits
#[test]
fn truncated_input_never_decodes_long() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut buffer = [0u8; MAX_INSTR_LEN];

    for _ in 0..50_000 {
        for byte in &mut buffer {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        for len in 0..buffer.len() {
            if let Some(insn) = decode(&buffer[..len]) {
                assert!(insn.len <= len, "claimed {} bytes of {len}", insn.len);
            }
        }
    }
}

/// Every prefix of a valid encoding is missing part of an immediate
#[test]
fn every_proper_prefix_of_an_instruction_is_short_or_invalid() {
    let encodings: &[&[u8]] = &[
        &[0x41, 0x80, 0x80, 0x80, 0x80, 0x78],
        &[0x43, 0x00, 0x00, 0x80, 0x3f],
        &[0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f],
        &[0x0e, 0x02, 0x00, 0x01, 0x03],
        &[0x28, 0x02, 0x10],
        &[0x11, 0x02, 0x01],
        &[
            0xfd, 0x0c, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ],
    ];

    for encoding in encodings {
        let full = decode(encoding).expect("the whole encoding decodes");
        assert_eq!(full.len, encoding.len());

        for cut in 1..encoding.len() {
            let short = decode(&encoding[..cut]);
            assert!(
                short.is_none_or(|insn| insn.len < full.len),
                "{:02x?} decoded as a full instruction from {cut} bytes",
                encoding
            );
        }
    }
}

/// Anything that decodes is an operator this build knows how to name
#[test]
fn assigned_opcodes_all_have_names() {
    let mut buffer = [0u8; MAX_INSTR_LEN];

    for prefix in 0u8..=0xff {
        for second in 0u8..=0xff {
            buffer.fill(0);
            buffer[0] = prefix;
            buffer[1] = second;
            if let Some(insn) = decode(&buffer) {
                assert_ne!(insn.mnemonic(), "(unknown)", "{prefix:#04x} {second:#04x}");
            }
        }
    }
}

/// The operators legal only inside a particular block still have to decode, since disassembly
/// starts wherever the user is looking
#[test]
fn block_scoped_operators_decode_out_of_context() {
    for (encoding, expected) in [
        (&[0x05][..], "else"),
        (&[0x0b], "end"),
        (&[0x07, 0x00], "catch"),
        (&[0x19], "catch_all"),
        (&[0x18, 0x00], "delegate"),
    ] {
        let insn = decode(encoding).unwrap_or_else(|| panic!("{expected} should decode"));
        assert_eq!(insn.mnemonic(), expected);
    }
}
