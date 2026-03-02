
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import re

from final_pipline import run_pipeline

# FIXED PATHS
MODEL_PATH = r"D:\Liver Tumor Segmentation\Final Code\VNet\dataset\vnet_full_model.h5"
SEGMENT_DIR = r"E:\Lits challange dataset\validation"


class LiverTumorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Liver Tumor Segmentation Tool")

        # UPDATED WINDOW SIZE
        self.root.geometry("1280x820")
        self.root.resizable(False, False)

        self.ct_file = None
        self.seg_file = None

        self.setup_background()
        self.create_header()
        self.create_upload_frame()
        self.create_footer_buttons()

    # ---------------------------------------------------------
    # BACKGROUND
    # ---------------------------------------------------------
    def setup_background(self):
        try:
            bg = Image.open(r"D:\Liver Tumor Segmentation\Final Code\wallpaper 8.jpg")
            bg = bg.resize((1280, 820))
            self.bg_photo = ImageTk.PhotoImage(bg)
            tk.Label(self.root, image=self.bg_photo).place(x=0, y=0, relwidth=1, relheight=1)
        except:
            self.root.config(bg="lightblue")

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------
    def create_header(self):
        tk.Label(
            self.root,
            text="Liver Tumor Segmentation Tool\nBy: Aqib Mehmood",
            font=("Arial", 30, "bold"),
            bg="#083454",
            fg="white",
            pady=20
        ).pack(fill="x")

    # ---------------------------------------------------------
    # MAIN UPLOAD FRAME
    # ---------------------------------------------------------
    def create_upload_frame(self):
        self.frame = tk.Frame(self.root, bg="white", bd=4, relief="ridge")
        self.frame.place(relx=0.5, rely=0.45, anchor="center", width=550, height=300)

        tk.Label(
            self.frame,
            text="Upload CT Volume (NII):",
            font=("Arial", 18, "bold"),
            bg="white"
        ).pack(pady=25)

        tk.Button(
            self.frame,
            text="Browse File",
            width=15,
            font=("Arial", 14, "bold"),
            command=self.upload_ct,
            bg="#0F6DB8",
            fg="white"
        ).pack(pady=10)

        tk.Button(
            self.frame,
            text="Run Segmentation",
            font=("Arial", 16, "bold"),
            bg="#009944",
            fg="white",
            width=20,
            height=1,
            command=self.run_process
        ).pack(pady=25)

    # ---------------------------------------------------------
    # FOOTER
    # ---------------------------------------------------------
    def create_footer_buttons(self):
        tk.Button(
            self.root,
            text="Developer Info",
            font=("Arial", 12, "bold"),
            bg="#A60000",
            fg="white",
            command=self.show_developer
        ).place(x=20, y=770)

    # ---------------------------------------------------------
    # FILE SELECTOR
    # ---------------------------------------------------------
    def upload_ct(self):
        path = filedialog.askopenfilename(filetypes=[("NIfTI Files", "*.nii *.nii.gz")])
        if not path:
            return

        self.ct_file = path

        match = re.search(r"volume-(\d+)", os.path.basename(path))
        if not match:
            messagebox.showerror("Invalid File", "Filename must contain 'volume-XX.nii' format.")
            return

        vol_num = match.group(1)
        self.seg_file = os.path.join(SEGMENT_DIR, f"segmentation-{vol_num}.nii")

        messagebox.showinfo("File Loaded", f"CT File Loaded:\n{self.ct_file}")

    # ---------------------------------------------------------
    # RUN PIPELINE
    # ---------------------------------------------------------
    def run_process(self):
        if not self.ct_file:
            messagebox.showerror("Missing File", "Please upload CT volume first.")
            return

        self.show_progress()
        threading.Thread(target=self.execute_pipeline).start()

    def execute_pipeline(self):
        try:
            imgs = run_pipeline(
                self.ct_file,
                self.seg_file,
                MODEL_PATH,
                progress_callback=self.update_progress
            )
            self.root.after(0, lambda: self.show_results(imgs))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Segmentation Completed!"))

        except Exception as e:
            err = str(e)
            self.root.after(0, lambda err=err: messagebox.showerror("Pipeline Error", err))


        finally:
            self.progress_win.destroy()

    # ---------------------------------------------------------
    # PROGRESS WINDOW
    # ---------------------------------------------------------
    def show_progress(self):
        self.progress_win = tk.Toplevel(self.root)
        self.progress_win.title("Processing...")
        self.progress_win.geometry("350x150")
        self.progress_win.resizable(False, False)

        # Center the window
        self.progress_win.update_idletasks()
        x = self.root.winfo_x() + (1280 // 2) - (350 // 2)
        y = self.root.winfo_y() + (820 // 2) - (150 // 2)
        self.progress_win.geometry(f"+{x}+{y}")

        self.progress_label = tk.Label(
            self.progress_win,
            text="Starting...",
            font=("Arial", 15)
        )
        self.progress_label.pack(pady=40)

    def update_progress(self, text):
        self.progress_label.config(text=text)
        self.progress_label.update()

    # ---------------------------------------------------------
    # RESULTS WINDOW
    # ---------------------------------------------------------
    def show_results(self, imgs):
        win = tk.Toplevel(self.root)
        win.title("Segmentation Results")
        win.geometry("1100x500")

        for i, (name, array) in enumerate(imgs.items()):
            img = Image.fromarray(array).resize((350, 350))
            img = ImageTk.PhotoImage(img)

            tk.Label(win, text=name.upper(), font=("Arial", 14, "bold")).grid(row=0, column=i, pady=10)
            lbl = tk.Label(win, image=img)
            lbl.image = img
            lbl.grid(row=1, column=i, padx=20)

    # ---------------------------------------------------------
    # DEVELOPER INFO
    # ---------------------------------------------------------
    def show_developer(self):
        messagebox.showinfo(
            "Developer Info",
            "Developer: Aqib Mehmood\n"
            "Specialization: AI / ML / Medical Imaging\n"
            "Project: Liver Tumor Segmentation (3D VNet)"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    LiverTumorGUI(root)
    root.mainloop()
