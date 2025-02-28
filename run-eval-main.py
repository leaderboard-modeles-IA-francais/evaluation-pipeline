import subprocess
import os

def main():
    subprocess.run("run-eval-main.sh", shell=True, executable="/bin/bash")

if __name__ == "__main__":
    main()
