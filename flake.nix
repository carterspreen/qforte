{
  description = "qForte Statevector Simulator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:

    let
      pkgs = import nixpkgs { inherit system; };

      packages = with pkgs; [

        clang-tools
        pyright

        git
        tree
        tmux
        htop
        curl

        gh
        eza
        bat
        btop
        wget

        vim
        neovim
        neofetch
        starship
        ripgrep

        wl-clipboard
        xclip

      ];

      mambaRootPrefix = "$HOME/.local/share/mamba";

      defaultMambaEnv = "qforte-default-env";
      qiskitMambaEnv = "qforte-qiskit-env";

      defaultPythonVer = "python=3.8";
      qiskitPythonVer = "python=3.10";

      defaultPythonPkgs = pkgs.lib.concatStringsSep " " [
        "openblas"
        "psi4"
        "cmake"
        "pytest"
      ];

      qiskitPythonPkgs = pkgs.lib.concatStringsSep " " [
        "qiskit"
        "qiskit-qasm3-import"
        "qiskit-ibm-runtime"
        "qiskit-aer"
      ];

      extraPythonPkgs = pkgs.lib.concatStringsSep " " [
        "matplotlib"
        "pandas"
        "sympy"
      ];

      allPythonPkgs = pkgs.lib.concatStringsSep " " [
        defaultPythonPkgs
        qiskitPythonPkgs
        extraPythonPkgs
      ];

      fhsDev = { mambaEnv ? defaultMambaEnv, pythonVer ? defaultPythonVer, pythonPkgs ? defaultPythonPkgs }:
        pkgs.buildFHSEnv {

        name = "qforte-dev";

        targetPkgs = pkgs: with pkgs; [
          micromamba
          gnumake
          gcc

          bash

          zsh
          zsh-completions
          zsh-syntax-highlighting
          zsh-autosuggestions
        ];

        runScript = "zsh --login";

        profile = ''
          set -e

          if [ "$0" = zsh ]; then
            eval "$(micromamba shell hook --shell=zsh)"
          elif [ "$0" = bash ]; then
            eval "$(micromamba shell hook --shell=bash)"
          else
            eval "$(micromamba shell hook --shell=posix)"
          fi

          export MAMBA_ROOT_PREFIX="${mambaRootPrefix}"
  
          if [ ! -d "$MAMBA_ROOT_PREFIX/envs/${mambaEnv}" ]; then
            micromamba create -n ${mambaEnv} -y -c conda-forge ${pythonVer} ${pythonPkgs}
          fi

          micromamba activate ${mambaEnv}

          set +e
        '';

      };

      fhsBuild = { mambaEnv ? defaultMambaEnv, pythonVer ? defaultPythonVer, pythonPkgs ? defaultPythonPkgs }:

        pkgs.buildFHSEnv {

          name = "qforte-build";

          targetPkgs = pkgs: with pkgs; [
            micromamba
            gnumake
            gcc
          ];

          runScript = ''

            eval "$(micromamba shell hook --shell=posix)"
            export MAMBA_ROOT_PREFIX="${mambaRootPrefix}"

            if [ ! -d "$MAMBA_ROOT_PREFIX/envs/${mambaEnv}" ]; then
              micromamba create -n "${mambaEnv}" -y -c conda-forge ${pythonVer} ${pythonPkgs}
            fi

            micromamba activate "${mambaEnv}"
            ./scripts/sh/build-mamba.sh

          '';

        };

    in rec {

      devShells.shell-default = pkgs.mkShell {

        shellHook  = ''
          echo "starting qforte dev shell..."
          exec ${(fhsDev {}).out}/bin/qforte-dev;
        '';
        inherit packages;

      };

      devShells.shell-qiskit = pkgs.mkShell {

        shellHook  = ''
          echo "starting qforte dev shell..."
          exec ${(fhsDev {mambaEnv = qiskitMambaEnv; pythonVer = qiskitPythonVer; pythonPkgs = allPythonPkgs;}).out}/bin/qforte-dev;
        '';

        inherit packages;

      };

      devShells.build-default = pkgs.mkShell {

        shellHook  = ''
          echo "building and installing qforte..."
          exec ${(fhsBuild {}).out}/bin/qforte-build
        '';

      };

      devShells.build-qiskit = pkgs.mkShell {
        
        shellHook  = ''
          echo "building and installing qforte..."
          exec ${(fhsBuild {mambaEnv = qiskitMambaEnv; pythonVer = qiskitPythonVer; pythonPkgs = allPythonPkgs;}).out}/bin/qforte-build;
        '';

      };

      devShells.default = devShells.shell-qiskit;
      devShells.build = devShells.build-qiskit;

    }
  );
}

