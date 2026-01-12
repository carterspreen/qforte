{ pkgs ? import <nixpkgs> {} }:
let
  fhs = pkgs.buildFHSEnv {
    name = "qforte-qiskit";
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

      if [ ! -d "$MAMBA_ROOT_PREFIX/envs/qforte-qiskit-env" ]; then
        micromamba create -n qforte-qiskit-env -y -c conda-forge python=3.10 openblas psi4 \
        qiskit qiskit-ibm-runtime qiskit-aer \
        cmake pytest
      fi

      micromamba activate qforte-qiskit-env

      set +e
    '';
  };

  projectRoot = toString ../..;
in

pkgs.mkShell {
  name = "qforte-qiskit-shell";

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
    set -e
    cd ${projectRoot} || exit 1
    ln -sf scripts/sh/build-mamba.sh build.sh
    ln -sf scripts/nix/shell-qiskit.nix shell.nix
    echo "launching FHS dev shell in $(pwd) …"
    exec ${fhs.out}/bin/qforte-qiskit
  '';
}
