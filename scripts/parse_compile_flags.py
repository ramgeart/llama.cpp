import argparse
import os
import re
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--github-env", required=True)
    args = parser.parse_args()

    config_path = "compile_flags.conf"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        sys.exit(1)

    known_keys = {
        "BUILD_TYPE", "BUILD_DIR", "BUILD_TARGETS", "BUILD_JOBS",
        "LINUX_GENERATOR", "WINDOWS_GENERATOR",
        "COMMON_CMAKE_OPTIONS", "CUDA_CMAKE_OPTIONS", "NOCUDA_CMAKE_OPTIONS",
        "CUDA_VERSION", "LINUX_APT_PACKAGES", "WINDOWS_CHOCO_PACKAGES",
        "ARTIFACT_PREFIX", "RUN_TESTS", "LINUX_TEST_COMMAND", "WINDOWS_TEST_COMMAND",
        "CUDALINUX_COMPILE", "CUDAWIN_COMPILE", "NOCUDALINUX_COMPILE", "NOCUDAWIN_COMPILE"
    }

    config = {}
    with open(config_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                print(f"Error: Malformed line {line_num}: {line}")
                sys.exit(1)

            key, value = line.split("=", 1)
            key = key.strip()

            if not re.match(r"^[A-Z0-9_]+$", key):
                print(f"Error: Invalid key format at line {line_num}: {key}")
                sys.exit(1)

            if key not in known_keys:
                print(f"Error: Unknown key at line {line_num}: {key}")
                sys.exit(1)

            if key in config:
                print(f"Error: Duplicate key at line {line_num}: {key}")
                sys.exit(1)

            if any(illegal in value for illegal in ["`", "$(", "${"]):
                print(f"Error: Illegal characters in value at line {line_num}")
                sys.exit(1)

            config[key] = value

    target = args.target.lower()
    target_flag_map = {
        "cudalinux": "CUDALINUX_COMPILE",
        "cudawin": "CUDAWIN_COMPILE",
        "nocudalinux": "NOCUDALINUX_COMPILE",
        "nocudawin": "NOCUDAWIN_COMPILE"
    }

    if target not in target_flag_map:
        print(f"Error: Unknown target {target}")
        sys.exit(1)

    compile_flag_key = target_flag_map[target]
    raw_val = config.get(compile_flag_key, "FALSE").strip().lower()

    bool_map = {
        "true": "true", "1": "true", "yes": "true",
        "false": "false", "0": "false", "no": "false"
    }

    if raw_val not in bool_map:
        print(f"Error: Invalid boolean value for {compile_flag_key}: {raw_val}")
        sys.exit(1)

    target_compile = bool_map[raw_val]

    is_cuda = target.startswith("cuda")
    backend_options = config.get("CUDA_CMAKE_OPTIONS" if is_cuda else "NOCUDA_CMAKE_OPTIONS", "")

    with open(args.github_env, "a") as env_file:
        for k, v in config.items():
            env_file.write(f"{k}={v}\n")
        env_file.write(f"TARGET_COMPILE={target_compile}\n")
        env_file.write(f"BACKEND_CMAKE_OPTIONS={backend_options}\n")

    print(f"Target {target} parsed successfully. TARGET_COMPILE={target_compile}")

if __name__ == "__main__":
    main()
