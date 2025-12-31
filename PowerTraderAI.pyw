#!/usr/bin/env python
"""
PowerTraderAI Launcher
Double-click this file to start PowerTraderAI Hub without a console window.
"""

import os
import sys
import warnings
import subprocess

# Set to True to enable debug logging to launcher_debug.log
DEBUG = False

# Create a debug log to help troubleshoot startup issues
debug_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launcher_debug.log")
def log_debug(message):
    if DEBUG:
        with open(debug_log, "a") as f:
            f.write(f"{message}\n")

log_debug("=== PowerTraderAI Launcher Starting ===")

# Suppress the pkg_resources deprecation warning from kucoin
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Ensure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Check if required packages are installed, install if needed
def check_and_install_requirements():
    """Check if key packages are installed, install requirements.txt if not."""
    log_debug("Starting requirement check...")
    log_debug(f"Python executable: {sys.executable}")
    log_debug(f"Python version: {sys.version}")
    
    # First, verify tkinter is available
    try:
        log_debug("Importing tkinter...")
        import tkinter as tk
        from tkinter import messagebox, scrolledtext
        log_debug("tkinter imported successfully")
    except ImportError as e:
        log_debug(f"ERROR: tkinter import failed: {e}")
        print(f"ERROR: tkinter is not available: {e}")
        print("tkinter should come with Python. Please reinstall Python.")
        input("Press Enter to exit...")
        sys.exit(1)
    
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
        log_debug("Creating dialog...")
        
        # Create root window as the dialog (don't hide it!)
        root = tk.Tk()
        root.title("First-Time Setup")
        root.geometry("420x200")
        root.resizable(False, False)
        log_debug("Root created")
        
        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        # Message
        missing_list = "\n".join(f"  • {pkg}" for pkg in missing_packages)
        log_debug(f"Package list: {missing_list}")
        msg = tk.Label(
            root,
            text="PowerTraderAI needs to install required packages.\n\n"
                 "This is a one-time setup and may take a few minutes.\n\n"
                 "Alternatively, run: pip install -r requirements.txt",
            justify=tk.CENTER,
            padx=20,
            pady=15,
            font=("TkDefaultFont", 9)
        )
        msg.pack()
        
        # Button frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        result = {"install": False}
        
        def on_install():
            result["install"] = True
            root.quit()
        
        def on_cancel():
            result["install"] = False
            root.quit()
        
        tk.Button(btn_frame, text="Install", command=on_install, width=10, default=tk.ACTIVE).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        log_debug("Showing dialog...")
        root.mainloop()
        log_debug("User responded")
        
        install_needed = result["install"]
        log_debug(f"Install chosen: {install_needed}")
        
        if install_needed:
            log_debug("Starting installation...")
            requirements_path = os.path.join(script_dir, "requirements.txt")
            if not os.path.isfile(requirements_path):
                messagebox.showerror(
                    "Error",
                    f"requirements.txt not found in:\n{script_dir}"
                )
                root.destroy()
                sys.exit(1)
            
            try:
                # Reuse the root window for installation progress
                # Clear existing widgets
                for widget in root.winfo_children():
                    widget.destroy()
                
                root.title("Installing Packages")
                root.geometry("600x400")
                
                tk.Label(
                    root,
                    text="Installing required packages...",
                    font=("TkDefaultFont", 10, "bold"),
                    padx=10,
                    pady=10
                ).pack()
                
                # Text widget to show output
                output_text = scrolledtext.ScrolledText(
                    root,
                    width=90,
                    height=25,
                    wrap=tk.NONE,
                    font=("Courier", 8)
                )
                output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
                output_text.insert(tk.END, f"Installing from: {requirements_path}\n\n")
                root.update()
                log_debug("Progress window ready")
                
                # Install packages and show output
                log_debug("Starting pip install...")
                process = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install", "-r", requirements_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Read output line by line
                for line in process.stdout:
                    output_text.insert(tk.END, line)
                    output_text.see(tk.END)
                    root.update()
                
                process.wait()
                log_debug(f"pip install completed with code: {process.returncode}")
                
                if process.returncode == 0:
                    output_text.insert(tk.END, "\n✅ Installation completed successfully!")
                    output_text.see(tk.END)
                    root.update()
                    log_debug("Installation successful")
                    
                    # Keep window open for 2 seconds so user can see success
                    root.after(2000, root.destroy)
                    root.mainloop()
                    
                    # Clear the missing packages list - trust that pip installed them
                    # (They won't be importable in the same process, but will work on next launch)
                    missing_packages.clear()
                    log_debug("Cleared missing packages list")
                else:
                    log_debug("Installation failed")
                    raise subprocess.CalledProcessError(process.returncode, process.args)
                    
            except subprocess.CalledProcessError as e:
                log_debug(f"CalledProcessError: {e}")
                try:
                    root.destroy()
                except:
                    pass
                messagebox.showerror(
                    "Installation Failed",
                    f"Failed to install required packages.\n\n"
                    f"Please run this command manually:\n"
                    f"python -m pip install -r requirements.txt\n\n"
                    f"Error: {e}"
                )
                sys.exit(1)
            except Exception as e:
                log_debug(f"Exception during install: {e}")
                try:
                    root.destroy()
                except:
                    pass
                messagebox.showerror(
                    "Error",
                    f"An error occurred during installation:\n\n{e}"
                )
                sys.exit(1)
        else:
            log_debug("User cancelled installation")
            try:
                root.destroy()
            except:
                pass
            sys.exit(0)

# Check requirements before importing pt_hub
try:
    check_and_install_requirements()
except Exception as e:
    import traceback
    print(f"ERROR during requirement check: {e}")
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)

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
    try:
        app = pt_hub.PowerTraderHub()
        app.mainloop()
    except Exception as e:
        import traceback
        print(f"ERROR running PowerTraderAI Hub: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)
