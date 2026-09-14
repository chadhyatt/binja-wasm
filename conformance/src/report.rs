//! What the sweep found

use std::collections::BTreeMap;
use std::path::Path;

pub struct Report {
    pub files: usize,
    pub modules: usize,
    pub functions: usize,
    pub instructions: u64,
    /// Instructions longer than an architecture plugin is allowed to claim
    pub oversized: usize,
    /// Bodies the harness gave up on part way through, whose remaining checks did not run
    pub abandoned: usize,
    /// Modules the validator rejected outright, so nothing in them was compared
    pub invalid: usize,
    /// Instructions the validator had no arity for, so neither check ran on them
    pub unchecked: u64,
    /// Operators that fall through but whose stack effect this crate cannot state, the real gaps
    pub unmodelled: BTreeMap<String, u64>,
    /// Operators with no stack effect to state because control leaves them
    pub leaves: BTreeMap<String, u64>,
    /// Heights compared against the validator's, and the ones recovery could not state; the second
    /// is what a branch out of that stretch cannot unwind correctly, so it is the one to watch
    pub heights_checked: u64,
    pub heights_unknown: u64,
    /// Branch edges the lifter has to unwind, each one a stack pointer the core would otherwise
    /// fail to merge
    pub unwinding: u64,
    pub failures: Vec<String>,
    /// Corpus files with forms nothing could be read out of, which is a hole rather than a pass
    pub unreadable: Vec<String>,
    /// The ones already known to be unreadable, listed so the count staying put is visible
    pub expected_unreadable: Vec<String>,
    /// Listed as unreadable but read in full, so the list is out of date
    pub stale_expectations: Vec<String>,
    /// Failures past this many are counted but not kept
    limit: usize,
    dropped: usize,
}

impl Report {
    pub fn new(limit: usize) -> Self {
        Report {
            files: 0,
            modules: 0,
            functions: 0,
            instructions: 0,
            oversized: 0,
            abandoned: 0,
            invalid: 0,
            unchecked: 0,
            unmodelled: BTreeMap::new(),
            leaves: BTreeMap::new(),
            heights_checked: 0,
            heights_unknown: 0,
            unwinding: 0,
            failures: Vec::new(),
            unreadable: Vec::new(),
            expected_unreadable: Vec::new(),
            stale_expectations: Vec::new(),
            limit,
            dropped: 0,
        }
    }

    /// A sweep that checked nothing is not a pass, or anything stopping modules reaching the
    /// checks reads like a clean run
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
            && self.unreadable.is_empty()
            && self.stale_expectations.is_empty()
            && self.modules > 0
            && self.functions > 0
            && self.instructions > 0
    }

    pub fn fail(&mut self, file: &Path, detail: impl std::fmt::Display) {
        if self.failures.len() < self.limit {
            self.failures.push(format!("{}: {detail}", file.display()));
        } else {
            self.dropped += 1;
        }
    }

    pub fn unmodelled(&mut self, mnemonic: String) {
        *self.unmodelled.entry(mnemonic).or_default() += 1;
    }

    pub fn leaves(&mut self, mnemonic: String) {
        *self.leaves.entry(mnemonic).or_default() += 1;
    }

    pub fn unwound(&mut self, known: bool) {
        if known {
            self.heights_checked += 1;
        } else {
            self.heights_unknown += 1;
        }
    }

    pub fn print(&self, quiet: bool, all: bool) {
        println!(
            "{} files, {} modules, {} functions, {} instructions",
            self.files, self.modules, self.functions, self.instructions
        );

        let unmodelled: u64 = self.unmodelled.values().sum();
        let leaves: u64 = self.leaves.values().sum();
        let modelled = self.instructions.saturating_sub(unmodelled + leaves);

        // Instructions control leaves are not something to model, so counting them would make an
        // honest `None` look like a bug
        let measurable = self.instructions.saturating_sub(leaves);
        let share = if measurable == 0 {
            0.0
        } else {
            modelled as f64 * 100.0 / measurable as f64
        };
        println!("stack effect modelled for {modelled} of {measurable} measurable ({share:.2}%)");
        println!(
            "{leaves} of {} instructions have no stack effect to model, control leaving them",
            self.instructions
        );
        println!(
            "operand stack height matched the validator on {} instructions, unknown on {}",
            self.heights_checked, self.heights_unknown
        );
        println!("{} branch edges unwind the operand stack", self.unwinding);
        if self.unchecked != 0 {
            println!(
                "{} of {} instructions had nothing to compare against",
                self.unchecked, self.instructions
            );
        }
        if self.invalid != 0 {
            println!(
                "{} modules rejected by the validator, so unchecked",
                self.invalid
            );
        }
        if self.abandoned != 0 {
            println!(
                "{} of {} bodies abandoned part way through, so their later checks did not run",
                self.abandoned, self.functions
            );
        }
        if self.oversized != 0 {
            println!(
                "{} instructions too long for any architecture plugin to claim",
                self.oversized
            );
        }

        if !quiet && !self.unmodelled.is_empty() {
            let mut worst: Vec<_> = self.unmodelled.iter().collect();
            worst.sort_by_key(|(name, count)| (std::cmp::Reverse(**count), (*name).clone()));
            let shown = if all { worst.len() } else { 15 };
            println!("\nunmodelled, most common first:");
            for (name, count) in worst.iter().take(shown) {
                println!("  {name:28} {count}");
            }
            if worst.len() > shown {
                println!(
                    "  ... and {} more, pass --all to see them",
                    worst.len() - shown
                );
            }
        }

        if !self.expected_unreadable.is_empty() && !quiet {
            println!(
                "\n{} files with forms skipped, each for a known reason:",
                self.expected_unreadable.len()
            );
            for file in &self.expected_unreadable {
                println!("  {file}");
            }
        }

        if !self.unreadable.is_empty() {
            println!(
                "\n{} files with forms nothing could be read out of:",
                self.unreadable.len()
            );
            for file in &self.unreadable {
                println!("  {file}");
            }
        }

        if !self.stale_expectations.is_empty() {
            println!(
                "\n{} files read in full but listed as unreadable, so drop them from the list:",
                self.stale_expectations.len()
            );
            for file in &self.stale_expectations {
                println!("  {file}");
            }
        }

        if self.modules == 0 || self.functions == 0 || self.instructions == 0 {
            println!("\nswept nothing, which is not a pass");
        }

        let total = self.failures.len() + self.dropped;
        if total == 0 {
            if self.passed() {
                println!("\nno failures");
            }
            return;
        }

        println!("\n{total} failures:");
        if quiet {
            return;
        }
        for failure in &self.failures {
            println!("  {failure}");
        }
        if self.dropped != 0 {
            println!("  ... and {} more, pass --all to see them", self.dropped);
        }
    }
}
