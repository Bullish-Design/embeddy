{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = [ 
    pkgs.git 
    pkgs.uv
    ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
  languages = {
      python = {
          enable = true;
          version = "3.13";
          venv.enable = true;
          uv.enable = true;
        };
    };

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  # devman — the automation plane (CONCEPT.md §5). `base` alone: this repository
  # ships no scheduled work and writes none of its own files.
  devman = {
    enable = true;
    project = "embeddy";
    groups = [ "base" ];
  };

  # https://devenv.sh/tasks/
  #
  # The two task names the `base` group calls (groups/base/README.md). devenv
  # owns each implementation; Dagu owns the composition (§6).
  #
  # The root pyproject is a virtual workspace: its dev deps are a uv
  # `[dependency-groups]`, not an extra, so the flag is `--group dev` and never
  # `--extra dev`. `qdrant_client` (and numpy) live there; the suite fails on
  # `ModuleNotFoundError: qdrant_client` without it. numpy's wheel needs libz at
  # runtime, which the devenv shell does not put on the loader path — hence the
  # `LD_LIBRARY_PATH` prefix.
  tasks = {
    "embeddy:lint".exec = "uv run --group dev ruff check .";
    "embeddy:test".exec = "LD_LIBRARY_PATH=${pkgs.zlib}/lib:\${LD_LIBRARY_PATH:-} uv run --group dev pytest";

    "base:check".after = [ "embeddy:lint" ];
    "base:test".after = [ "embeddy:test" ];
  };

  enterShell = ''
    hello
    git --version
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
