use clap::CommandFactory;

use crate::Cli;
use crate::CompletionsArgs;
use crate::util::Result;

/// Run `cargo xtask completions <shell>`: generate shell completion script.
///
/// Writes the completion script to stdout. The caller should redirect stdout
/// to the appropriate shell completion directory. No `Shell` parameter is
/// needed because this command does not invoke any external processes.
///
/// # Examples
///
/// ```text
/// cargo xtask completions bash >> ~/.bashrc
/// cargo xtask completions zsh > ~/.zfunc/_cargo-xtask
/// cargo xtask completions fish > ~/.config/fish/completions/cargo-xtask.fish
/// ```
pub(crate) fn run(args: CompletionsArgs) -> Result<()> {
    let mut cmd = Cli::command();
    // Use "cargo-xtask" as the binary name so completions trigger when typing
    // `cargo xtask <TAB>` after setting up the shell alias.
    clap_complete::generate(args.shell, &mut cmd, "cargo-xtask", &mut std::io::stdout());
    Ok(())
}
