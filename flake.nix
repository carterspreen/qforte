#begin nix flake
{
  #nix flake metadata
  description = "qForte Statevector Simulator";

  #nix flake inputs
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  #nix flake outputs
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:

    #support mac and linux on x86_64 and aarch64
    flake-utils.lib.eachDefaultSystem (
      system:

      #define variables for per-system attributes
      let
        #alias for nixpkgs repository
        pkgs = import nixpkgs { inherit system; };

        #development shell packages
        packages = with pkgs; [

          #fetchers

          #classic cli tools
          git
          vim
          tmux
          tree

          #modern cli tools
          eza
          bat
          dust
          neovim
          starship
          fastfetch

          #wayland clipboard integration
          wl-clipboard

        ];

        #location for mamba environments
        mambaRootPrefix = "$HOME/.local/share/mamba";

        #fallback python version
        defaultPythonVer = "python=3.13";
        #fallback mamba environment
        defaultMambaEnv = "qforte-default";

        #qiskit python version
        qiskitPythonVer = "python=3.13";
        #qiskit mamba env
        qiskitMambaEnv = "qforte-qiskit";

        #qforte base deps
        defaultPythonPkgs = pkgs.lib.concatStringsSep " " [
          "openblas"
          "psi4"
          "cmake"
          "pytest"
        ];

        #qforte qiskit deps
        qiskitPythonPkgs = pkgs.lib.concatStringsSep " " [
          "qiskit"
          "qiskit-qasm3-import"
          "qiskit-ibm-runtime"
          "qiskit-aer"
        ];

        #qforte optional deps
        extraPythonPkgs = pkgs.lib.concatStringsSep " " [
          "sympy"
        ];

        #bundle all deps
        allPythonPkgs = pkgs.lib.concatStringsSep " " [
          defaultPythonPkgs
          qiskitPythonPkgs
          extraPythonPkgs
        ];

        #function to set up qforte dev environment
        fhsDev =

          #function arguments
          {
            mambaEnv ? defaultMambaEnv,
            pythonVer ? defaultPythonVer,
            pythonPkgs ? defaultPythonPkgs,
          }:

          #function output is an fhs (needed for micromamba)
          pkgs.buildFHSEnv {

            #fhs metadata
            name = "qforte-dev";

            #fhs packages
            targetPkgs =
              pkgs: with pkgs; [
                zsh
                zsh-completions
                zsh-syntax-highlighting
                zsh-autosuggestions
                micromamba
              ];


            #fhs command (starts zsh with micromamba env initialized)
            runScript = pkgs.writeShellScript "startDevShell" ''

              set -euo pipefail
                
              #add host nix to path
              for p in "/nix/var/nix/profiles/default/bin" "$HOME/.nix-profile/bin"; do
                if [ -x "$p/nix" ]; then
                    export PATH="$p:$PATH"
                    break
                fi
              done

              #save zsh config location
              if [[ -v ZDOTDIR ]]; then
                  real_zdotdir=$ZDOTDIR
              else
                  real_zdotdir=""
              fi

              #make a temporary directory for the zsh wrapper
              wrapper="$(mktemp -d)"
              trap 'rm -rf $wrapper' EXIT INT TERM

              #wrapper script (runs before host .zshenv)
              cat > $wrapper/.zshenv << EOF

                #initialize micromamba
                export MAMBA_ROOT_PREFIX="${mambaRootPrefix}"
                if micromamba --version >/dev/null 2>&1; then
                    eval "\$(micromamba shell hook --shell=zsh)"
                    if [ ! -d "\$MAMBA_ROOT_PREFIX" ]; then
                      mkdir -p "\$MAMBA_ROOT_PREFIX"
                    fi
                    if [ ! -d "\$MAMBA_ROOT_PREFIX/envs/${mambaEnv}" ]; then
                      micromamba create -n ${mambaEnv} -y -c conda-forge ${pythonVer} ${pythonPkgs}
                    fi
                    micromamba activate ${mambaEnv}
                fi

                #restore saved zsh config location
                real_zdotdir=$real_zdotdir
                if [[ -z "\$real_zdotdir" ]]; then
                    unset ZDOTDIR
                    if [ -r "\$HOME/.zshenv" ]; then
                        source "\$HOME/.zshenv"
                    fi
                else
                    export ZDOTDIR=\$real_zdotdir
                    if [ -r "\$real_zdotdir/.zshenv" ]; then
                        source "\$real_zdotdir/.zshenv"
                    fi
                fi

              EOF

              #run zsh with the wrapper script
              export ZDOTDIR="$wrapper"
              exec zsh --login

            '';
            #fhs command (starts zsh with micromamba env initialized)
            #runScript = ''
            #  zsh -c "$(cat <<'EOF'
            #  echo 'eval "$(micromamba shell hook --shell=zsh)" && micromamba activate ${mambaEnv}' > /etc/zprofile
            #  zsh --login
            #  EOF
            #  )"
            #'';

            ##fhs profile (creates micromamba env if nonexistent)
            #profile = ''
            #  set -e
            #  eval "$(micromamba shell hook --shell=posix)"
            #  export MAMBA_ROOT_PREFIX="${mambaRootPrefix}"
            #  if [ ! -d "$MAMBA_ROOT_PREFIX" ]; then
            #    mkdir -p "$MAMBA_ROOT_PREFIX"
            #  fi
            #  if [ ! -d "$MAMBA_ROOT_PREFIX/envs/${mambaEnv}" ]; then
            #    micromamba create -n ${mambaEnv} -y -c conda-forge ${pythonVer} ${pythonPkgs}
            #  fi
            #'';

          };

        #function to set up a build/install environment
        fhsBuild =

          #function arguments
          {
            mambaEnv ? defaultMambaEnv,
            pythonVer ? defaultPythonVer,
            pythonPkgs ? defaultPythonPkgs,
          }:

          #function output is an fhs (needed for micromamba)
          pkgs.buildFHSEnv {

            #fhs metadata
            name = "qforte-build";

            #fhs packages
            targetPkgs =
              pkgs: with pkgs; [
                micromamba
                git
              ];

            #fhs command (exits cleanly)
            runScript = ''
              bash -c exit
            '';

            #fhs profile (builds qforte)
            profile = ''
              set -e

              eval "$(micromamba shell hook --shell=posix)"
              export MAMBA_ROOT_PREFIX="${mambaRootPrefix}"

              if [ ! -d "$MAMBA_ROOT_PREFIX" ]; then
                mkdir -p "$MAMBA_ROOT_PREFIX"
              fi
              if [ ! -d "$MAMBA_ROOT_PREFIX/envs/${mambaEnv}" ]; then
                micromamba create -n "${mambaEnv}" -y -c conda-forge ${pythonVer} ${pythonPkgs}
              fi

              CDPATH= cd $(git rev-parse --show-toplevel)

              micromamba activate "${mambaEnv}" || { echo "micromamba env not found"; exit 1; }

              cp -f CMakeLists.txt CMakeLists.txt.tmp || { echo "failed to backup CMakeLists.txt"; exit 1; }

              sed -i "s|set(CMAKE_PREFIX_PATH \".*\")|set(CMAKE_PREFIX_PATH \"$CONDA_PREFIX\")|" CMakeLists.txt

              if python setup.py develop; then
                  BUILD_EXIT=0; echo "build succeeded"
              else
                  BUILD_EXIT=2; echo "build failed"
              fi

              mv -f CMakeLists.txt.tmp CMakeLists.txt || { echo "failed to restore CMakeLists.txt"; exit 1; }

              exit $BUILD_EXIT

            '';

          };

        #sets up an environment for resetting environment
        fhsSetup =

          #function arguments
          {
            mambaEnv ? defaultMambaEnv,
            pythonVer ? defaultPythonVer,
            pythonPkgs ? defaultPythonPkgs,
          }:

          #function output is an fhs (needed for micromamba)
          pkgs.buildFHSEnv {

            #fhs metadata
            name = "qforte-setup";

            #fhs packages
            targetPkgs =
              pkgs: with pkgs; [
                micromamba
              ];

            #fhs command (exits cleanly)
            runScript = "bash -c exit";

            #fhs profile (resets mamba env)
            profile = ''

              set -e
              eval "$(micromamba shell hook --shell=posix)"
              export MAMBA_ROOT_PREFIX="${mambaRootPrefix}"

              if [ ! -d "$MAMBA_ROOT_PREFIX" ]; then
                mkdir -p "$MAMBA_ROOT_PREFIX"
              fi
              if [ -d "$MAMBA_ROOT_PREFIX/envs/${mambaEnv}" ]; then
                micromamba env remove -n "${mambaEnv}" -y
              fi
              micromamba create -n "${mambaEnv}" -y -c conda-forge ${pythonVer} ${pythonPkgs}
              exit 0

            '';

          };

      in
      #begin per-system attribute set (allowing recursive evaluation)
      rec {

        #qforte development shell with base deps
        devShells.shell-default = pkgs.mkShell {

          #exec into development environment
          shellHook = ''
            echo "starting qforte dev shell..."
            exec ${(fhsDev { }).out}/bin/qforte-dev;
          '';

          #inherit development packages
          inherit packages;

        };

        #qforte development shell with qiskit deps
        devShells.shell-qiskit = pkgs.mkShell {

          #exec into development environment
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

          #inherit development packages
          inherit packages;

        };

        #qforte build shell with base deps
        devShells.build-default = pkgs.mkShell {

          #exec into build shell
          shellHook = ''
            echo "building and installing qforte..."
            exec ${(fhsBuild { }).out}/bin/qforte-build
          '';

        };

        #qforte build shell with qiskit deps
        devShells.build-qiskit = pkgs.mkShell {

          #exec into build shell
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

        #qforte setup shell with base deps
        devShells.setup-default = pkgs.mkShell {

          #exec into setup shell
          shellHook = ''
            echo "building and installing qforte..."
            exec ${(fhsSetup { }).out}/bin/qforte-setup
          '';

        };

        #qforte setup shell with qiskit deps
        devShells.setup-qiskit = pkgs.mkShell {

          #exec into setup shell
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

        #example app to output the nix store path
        apps.storePath = {

          #app metadata
          type = "app";

          #app must run from the nix store
          program = "${pkgs.writeShellScript "myapp" ''
            echo -e "\nstore path: ${self}" && exit
          ''}";

        };

        #specify derivation for 'nix develop'
        devShells.default = devShells.shell-qiskit;

        #specify derivation for 'nix develop #build'
        devShells.build = devShells.build-qiskit;

        #specify derivation for 'nix develop #setup'
        devShells.setup = devShells.setup-qiskit;

      } # end per-system attribute set
    ); # end eachDefaultSystem
} # end flake
