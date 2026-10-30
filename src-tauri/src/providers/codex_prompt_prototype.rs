#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "finite private fixture probe; invoke through the documented isolated workflow"]
    fn private_batch_probe() {
        if let Err(reason) = super::super::prompt_state::run_private_probe() {
            panic!("prototype probe refused: {reason}");
        }
    }
}
