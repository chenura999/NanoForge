# NanoForge Makefile

.PHONY: all build run-daemon run-client clean check fmt

all: build

# Build both binaries
build:
	cd nanoforge && cargo build

# Set capabilities and run the daemon
# Uses sudo only for setcap
run-daemon: build
	sudo setcap cap_perfmon,cap_sys_admin+ep nanoforge/target/debug/daemon
	./nanoforge/target/debug/daemon

# Run the client application
run-client: build
	cd nanoforge && cargo run --bin nanoforge

# Run tests
test:
	cd nanoforge && cargo test

# Check code quality
check:
	cd nanoforge && cargo clippy
	cd nanoforge && cargo fmt --check

# Clean build artifacts
clean:
	cd nanoforge && cargo clean
