{
  description = "qForte Statevector Simulator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:

      let
        pkgs = import nixpkgs { inherit system; };

        packages = with pkgs; [

          #clang-tools
          #pyright

          git
          tree
          #htop
          #curl

          eza
          bat
          #btop
          #wget

          vim
          neovim
          starship
          #neofetch
          #ripgrep

          wl-clipboard
          #xclip

        ];

        mambaRootPrefix = "$HOME/.local/share/mamba";

        defaultMambaEnv = "qforte-default";
        qiskitMambaEnv = "qforte-qiskit";

        defaultPythonVer = "python=3.13";
        qiskitPythonVer = "python=3.13";

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
          "sympy"
        ];

        allPythonPkgs = pkgs.lib.concatStringsSep " " [
          defaultPythonPkgs
          qiskitPythonPkgs
          extraPythonPkgs
        ];

        fhsDev =
          {
            mambaEnv ? defaultMambaEnv,
            pythonVer ? defaultPythonVer,
            pythonPkgs ? defaultPythonPkgs,
          }:
          pkgs.buildFHSEnv {

            name = "qforte-dev";

            targetPkgs =
              pkgs: with pkgs; [
                micromamba
                #gnumake
                #gcc

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

        fhsBuild =
          {
            mambaEnv ? defaultMambaEnv,
            pythonVer ? defaultPythonVer,
            pythonPkgs ? defaultPythonPkgs,
          }:

          pkgs.buildFHSEnv {

            name = "qforte-build";

            targetPkgs =
              pkgs: with pkgs; [
                micromamba
                git
              ];

            runScript = ''

              CDPATH= cd $(git rev-parse --show-toplevel)
              eval "$(micromamba shell hook --shell=posix)"
              export MAMBA_ROOT_PREFIX="${mambaRootPrefix}"

              if [ ! -d "$MAMBA_ROOT_PREFIX/envs/${mambaEnv}" ]; then
                micromamba create -n "${mambaEnv}" -y -c conda-forge ${pythonVer} ${pythonPkgs}
              fi

              micromamba activate "${mambaEnv}"
              ./qiskit_api/scripts/mamba/build-mamba.sh
              exit

            '';

          };

        fhsSetup =
          {
            mambaEnv ? defaultMambaEnv,
            pythonVer ? defaultPythonVer,
            pythonPkgs ? defaultPythonPkgs,
          }:

          pkgs.buildFHSEnv {

            name = "qforte-setup";

            targetPkgs =
              pkgs: with pkgs; [
                micromamba
              ];

            runScript = ''

              eval "$(micromamba shell hook --shell=posix)"
              export MAMBA_ROOT_PREFIX="${mambaRootPrefix}"

              if [ -d "$MAMBA_ROOT_PREFIX/envs/${mambaEnv}" ]; then
                micromamba env remove -n "${mambaEnv}" -y
              fi
              micromamba create -n "${mambaEnv}" -y -c conda-forge ${pythonVer} ${pythonPkgs}

            '';

          };

      in
      rec {

        devShells.shell-default = pkgs.mkShell {

          shellHook = ''
            echo "starting qforte dev shell..."
            exec ${(fhsDev { }).out}/bin/qforte-dev;
          '';
          inherit packages;

        };

        devShells.shell-qiskit = pkgs.mkShell {

          shellHook = ''
            echo "starting qforte dev shell..."
            exec ${
              (fhsDev {
                mambaEnv = qiskitMambaEnv;
                pythonVer = qiskitPythonVer;
                pythonPkgs = allPythonPkgs;
              }).out
            }/bin/qforte-dev;
          '';

          inherit packages;

        };

        devShells.build-default = pkgs.mkShell {

          shellHook = ''
            echo "building and installing qforte..."
            exec ${(fhsBuild { }).out}/bin/qforte-build
          '';

        };

        devShells.build-qiskit = pkgs.mkShell {

          shellHook = ''
            echo "building and installing qforte..."
            exec ${
              (fhsBuild {
                mambaEnv = qiskitMambaEnv;
                pythonVer = qiskitPythonVer;
                pythonPkgs = allPythonPkgs;
              }).out
            }/bin/qforte-build;
          '';

        };

        devShells.setup-default = pkgs.mkShell {

          shellHook = ''
            echo "building and installing qforte..."
            exec ${(fhsSetup { }).out}/bin/qforte-setup
          '';

        };

        devShells.setup-qiskit = pkgs.mkShell {

          shellHook = ''
            echo "building and installing qforte..."
            exec ${
              (fhsSetup {
                mambaEnv = qiskitMambaEnv;
                pythonVer = qiskitPythonVer;
                pythonPkgs = allPythonPkgs;
              }).out
            }/bin/qforte-setup;
          '';

        };

        devShells.default = devShells.shell-qiskit;
        devShells.build = devShells.build-qiskit;
        devShells.setup = devShells.setup-qiskit;

      }
    );
}
