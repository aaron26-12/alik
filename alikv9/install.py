
import subprocess
import sys

PACKAGES = [
    "customtkinter",
    "groq",
    "edge-tts",
    "SpeechRecognition",
    "pyaudio",
    "pyserial",
    "pygame",
    "pillow",
    "numpy",
    "opencv-python",
    "mediapipe",
]

def run(cmd: str) -> None:
    print(f"▶ {cmd}")
    subprocess.check_call(cmd, shell=True)

print("=" * 68)
print(" ANIMATRÓNICO IA PRO — instalador")
print("=" * 68)
print("\nActualizando pip...")
run(f'"{sys.executable}" -m pip install --upgrade pip')

print("\nInstalando dependencias...")
for pkg in PACKAGES:
    try:
        run(f'"{sys.executable}" -m pip install {pkg}')
        print(f"  ✅ {pkg}")
    except Exception as e:
        print(f"  ⚠️  {pkg}: {e}")

print("\nVerificando imports básicos...")
checks = {
    "customtkinter": "customtkinter",
    "serial": "serial",
    "pygame": "pygame",
    "PIL": "PIL.Image",
    "numpy": "numpy",
    "cv2": "cv2",
    "mediapipe": "mediapipe",
}
all_ok = True
for label, mod in checks.items():
    try:
        __import__(mod)
        print(f"  ✅ {label}")
    except Exception:
        print(f"  ❌ {label}")
        all_ok = False

print("\nListo." if all_ok else "\nRevisá los módulos marcados con ❌.")
