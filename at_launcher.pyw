#!/usr/bin/env python
"""
Apollo Launcher
Double-click this file to start Apollo Trader without a console window.
"""

import os
import sys
import warnings
import subprocess
import time

# Set to True to enable debug logging to launcher_debug.log
DEBUG = False

# Set to False to show console windows for trainer/thinker/trader (useful for debugging output that doesn't reach logs)
HIDE_CONSOLE_WINDOWS = True

# Create a debug log to help troubleshoot startup issues
debug_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launcher_debug.log")
def log_debug(message):
    if DEBUG:
        with open(debug_log, "a") as f:
            f.write(f"{message}\n")

log_debug("=== Apollo Launcher Starting ===")

# Set Windows AppUserModelID early so taskbar shows our icon (not Python's)
try:
    import ctypes
    myappid = 'apollotrader.cryptoai.26'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    log_debug(f"Set AppUserModelID: {myappid}")
except Exception as e:
    log_debug(f"Failed to set AppUserModelID: {e}")

# Suppress the pkg_resources deprecation warning from kucoin
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Ensure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Single instance check
def check_single_instance():
    """Ensure only one instance of Apollo is running."""
    lock_file = os.path.join(script_dir, ".apollo.lock")
    
    # Check if lock file exists and is recent
    if os.path.exists(lock_file):
        try:
            with open(lock_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process is still running (Windows-compatible)
            import psutil
            if psutil.pid_exists(pid):
                try:
                    process = psutil.Process(pid)
                    # Check if it's actually a Python process
                    if 'python' in process.name().lower():
                        log_debug(f"Another instance already running (PID: {pid})")
                        return False
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Stale lock file - remove it
            log_debug(f"Removing stale lock file (PID: {pid})")
            os.remove(lock_file)
        except Exception as e:
            log_debug(f"Error checking lock file: {e}")
            # Remove potentially corrupted lock file
            try:
                os.remove(lock_file)
            except:
                pass
    
    # Create new lock file with our PID
    try:
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        log_debug(f"Created lock file with PID: {os.getpid()}")
        return True
    except Exception as e:
        log_debug(f"Failed to create lock file: {e}")
        return True  # Allow to proceed even if lock file creation fails

# Check if required packages are installed, install if needed
def check_and_install_requirements():
    """Check if key packages are installed, install requirements.txt if not."""
    log_debug("Starting requirement check...")
    log_debug(f"Python executable: {sys.executable}")
    log_debug(f"Python version: {sys.version}")
    
    missing_packages = []
    required_imports = [
        ('kucoin', 'kucoin'),
        ('robin_stocks', 'robin-stocks'),
        ('requests', 'requests'),
        ('cryptography', 'cryptography'),
    ]
    
    # Check each package individually
    log_debug("Checking packages...")
    for module_name, package_name in required_imports:
        try:
            __import__(module_name)
            log_debug(f"  {module_name}: OK")
        except ImportError:
            log_debug(f"  {module_name}: MISSING")
            missing_packages.append(package_name)
    
    # If DEBUG is enabled, force show installer dialog for testing
    if DEBUG and not missing_packages:
        log_debug("DEBUG mode: forcing installer test")
        missing_packages.append("(test mode - no packages missing)")
    
    log_debug(f"Missing packages: {missing_packages}")
    
    if missing_packages:
        log_debug("Starting auto-install...")
        
        # Auto-install in console (no GUI)
        print("\n" + "="*60)
        print("Apollo Trader - First Time Setup")
        print("="*60)
        print(r"""
    ▀█▀ █▀█   ▀█▀ █ █ █▀▀   █▀▄▀█ █▀█ █▀█ █▄ █ █
     █  █ █    █  █▀█ ██▄   █ ▀ █ █▄█ █▄█ █ ▀█ ▄
                       
        🚀  Crypto Trading AI  🚀
        """)
        print("Installing required packages:")
        for pkg in missing_packages:
            print(f"  • {pkg}")
        print("\nThis may take a few moments...\n")
        log_debug("Printed header")
        
        # Check for requirements.txt
        requirements_path = os.path.join(script_dir, "requirements.txt")
        if not os.path.isfile(requirements_path):
            log_debug(f"ERROR: requirements.txt not found at {requirements_path}")
            print(f"ERROR: requirements.txt not found in:\n  {script_dir}")
            input("\nPress Enter to exit...")
            sys.exit(1)
        
        try:
            # Run pip install with real-time console output
            log_debug("Starting pip install...")
            print(f"Running: {sys.executable} -m pip install -r requirements.txt\n")
            
            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "-r", requirements_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Show output line by line
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            log_debug(f"pip install completed with code: {process.returncode}")
            
            if process.returncode == 0:
                print("\n" + "="*60)
                print("✓ Installation completed successfully!")
                print("="*60)
                log_debug("Installation successful")
                
                # Fix .pyw file association so no console window appears on future launches
                print("\nConfiguring .pyw file association...")
                try:
                    fix_pyw_file_association()
                    print("✓ File association configured successfully!")
                    log_debug("File association fixed")
                except Exception as e:
                    log_debug(f"Failed to fix file association: {e}")
                    print(f"Note: Could not auto-configure .pyw association: {e}")
                
                print("\nSetup complete! Launching Apollo Trader...\n")
                import time
                time.sleep(2)  # Brief pause before continuing
                
                # Clear missing packages list so we can continue
                missing_packages.clear()
                log_debug("Cleared missing packages list")
            else:
                log_debug("Installation failed")
                print("\n" + "="*60)
                print("✗ Installation failed!")
                print("="*60)
                print("\nPlease run this command manually:")
                print(f"  python -m pip install -r requirements.txt")
                input("\nPress Enter to exit...")
                sys.exit(1)
                
        except Exception as e:
            log_debug(f"Exception during install: {e}")
            print(f"\nERROR: An error occurred during installation:\n  {e}")
            print("\nPlease run this command manually:")
            print(f"  python -m pip install -r requirements.txt")
            input("\nPress Enter to exit...")
            sys.exit(1)

def fix_pyw_file_association():
    """Fix .pyw file association to use pythonw.exe (no console window)."""
    try:
        import platform
        if platform.system() != 'Windows':
            return  # Only needed on Windows
        
        # Check if association is already correct
        import subprocess
        try:
            result = subprocess.run(['cmd', '/c', 'assoc', '.pyw'], 
                                   capture_output=True, text=True, timeout=5)
            current_assoc = result.stdout.strip()
            
            # If already associated, no need to fix
            if current_assoc and 'Python' in current_assoc:
                log_debug(f"PYW association already set: {current_assoc}")
                return
        except Exception:
            pass
        
        log_debug("Attempting to fix .pyw file association...")
        
        # Find pythonw.exe
        import shutil
        pythonw_path = shutil.which('pythonw.exe')
        
        if not pythonw_path:
            log_debug("pythonw.exe not found in PATH")
            return
        
        # Try to set the association (may fail without admin rights)
        try:
            # Set file type association
            subprocess.run(['cmd', '/c', 'assoc', '.pyw=Python.NoConFile'], 
                          capture_output=True, timeout=5, check=False)
            
            # Set command for the file type
            subprocess.run(['cmd', '/c', f'ftype', f'Python.NoConFile="{pythonw_path}" "%L" %*'],
                          capture_output=True, timeout=5, check=False)
            
            log_debug("Successfully set .pyw file association")
        except Exception as e:
            log_debug(f"Could not set file association (may need admin rights): {e}")
            
            # If automatic fix failed, create a helper batch file the user can run as admin
            try:
                bat_path = os.path.join(script_dir, "Fix_PYW_Association.bat")
                with open(bat_path, "w") as f:
                    f.write('@echo off\n')
                    f.write('echo Fixing .pyw file association...\n')
                    f.write('echo.\n')
                    f.write('assoc .pyw=Python.NoConFile\n')
                    f.write(f'ftype Python.NoConFile="{pythonw_path}" "%%L" %%*\n')
                    f.write('echo.\n')
                    f.write('echo Done! .pyw files will now run without a console window.\n')
                    f.write('echo.\n')
                    f.write('pause\n')
                log_debug(f"Created helper batch file at {bat_path}")
            except Exception as bat_error:
                log_debug(f"Could not create helper batch file: {bat_error}")
    except Exception as e:
        log_debug(f"Error in fix_pyw_file_association: {e}")
        # Silent fail - don't block app launch

# Check requirements before importing pt_hub
try:
    # First check if another instance is running
    if not check_single_instance():
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(
            "Apollo Already Running",
            "Another instance of Apollo is already running.\n\n"
            "Please close the existing instance before starting a new one."
        )
        root.destroy()
        sys.exit(0)
    
    check_and_install_requirements()
except Exception as e:
    import traceback
    print(f"ERROR during requirement check: {e}")
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)

# Set environment variable to control console window visibility
os.environ['POWERTRADER_HIDE_CONSOLE'] = '1' if HIDE_CONSOLE_WINDOWS else '0'

# Import and run the hub
try:
    import pt_hub
except Exception as e:
    import traceback
    print(f"ERROR importing pt_hub: {e}")
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)

if __name__ == "__main__":
    lock_file = os.path.join(script_dir, ".apollo.lock")
    try:
        app = pt_hub.ApolloHub()
        app.mainloop()
    except Exception as e:
        import traceback
        print(f"ERROR running Apollo Trader: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)
    finally:
        # Clean up lock file on exit
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                log_debug("Removed lock file on exit")
        except Exception as e:
            log_debug(f"Failed to remove lock file: {e}")
