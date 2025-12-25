import nanoforge
import time


def main():
    print("🧬 NanoForge Python Evolution Demo 🧬")
    print("====================================")

    # Simple script to evolve
    script = """
    fn main(n) {
        result = n + 1
        return result
    }
    """

    print(f"Input Script:\n{script}")

    print("🚀 Starting Evolution...")
    start_time = time.time()

    try:
        # Evolve: 10 generations, population 10
        # Returns (best_code, speedup)
        best_code, speedup = nanoforge.evolve(script, 10, 10)

        elapsed = time.time() - start_time
        print(f"\n✅ Evolution Complete in {elapsed:.2f}s")
        print(f"🏆 Speedup: {speedup:.2f}x")
        print(f"📄 Best Code:\n{best_code}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
