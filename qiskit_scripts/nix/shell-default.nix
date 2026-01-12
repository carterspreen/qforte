{ pkgs ? import <nixpkgs> {} }:
let
  fhs = pkgs.buildFHSEnv {
    name = "qforte-default";
    targetPkgs = pkgs: with pkgs; [
      micromamba
      gnumake
      gcc
    ];

    runScript = "bash --login";

    profile = ''
      set -e

      export MAMBA_ROOT_PREFIX="$PWD/.mamba"

      if [ "$0" = zsh ]; then
        eval "$(micromamba shell hook --shell=zsh)"
      elif [ "$0" = bash ]; then
        eval "$(micromamba shell hook --shell=bash)"
        eval "$(starship init bash)"
      else
        eval "$(micromamba shell hook --shell=posix)"
      fi

      if [ ! -d "$MAMBA_ROOT_PREFIX/envs/qforte-default-env" ]; then
        micromamba create -n qforte-default-env -y -c conda-forge python=3.8 openblas psi4 cmake pytest
      fi

      micromamba activate qforte-default-env

      set +e
    '';
  };

  projectRoot = toString ../..;

in

pkgs.mkShell {
  name = "qforte-default-shell";

  packages = with pkgs; [
    # TTY core
    vim
    tree
    eza
    bat
    dust
    btop
    neofetch

    gcc
    gdb
    lldb
    cmake
    clang
    clang-tools
    clang-analyzer
    clang-manpages

    pyright

    htop
    tmux
    git

    gh
    which

    zip
    unzip
    p7zip
    killall
    ripgrep
    wget


    neovim
    starship

    zsh
    zsh-completions
    zsh-syntax-highlighting
    zsh-autosuggestions
  ];

  shellHook = ''
    cd ${projectRoot} || exit 1
    ln -sf scripts/nix/shell-default.nix shell.nix
    ln -sf scripts/sh/build-mamba.sh build.sh
    echo "launching FHS dev shell in $(pwd) …"
    exec ${fhs.out}/bin/qforte-default
  '';
}
