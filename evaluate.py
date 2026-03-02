# """
# Diogo Amorim, 2018-07-10
# Evaluate Predictions - Vnet
# """

# import h5py
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def load_dataset(path, h5=True):
#     #print("Loading dataset... Shape:")
#     file = h5py.File(path, 'r')
#     if h5:
#         data = file.get('data')
#         truth = file.get('truth')
#     else:
#         data = file.get('data').value
#         truth = file.get('truth').value
#     # print(data.shape)
#     return data, truth

# def load_predictions(path):
#     path = os.path.join(path, 'predictions_vnet.h5')
#     print("Loading predictions... Shape:")
#     data = h5py.File(path, 'r')
#     predictions = data['predictions'][:]
#     print(predictions.shape)
#     return predictions


# def print_prediction(test, pred, m, slice):
#     fig = plt.figure()
#     y = fig.add_subplot(1, 2, 1)
#     y.imshow(test[m, :, :, slice], cmap='gray')
#     y = fig.add_subplot(1, 2, 2)
#     y.imshow(pred[m, :, :, slice], cmap='gray')
#     plt.show()


# save_dir = r"D:\SAM SEGMENTATION MASK\repo 6 bamf liver tumor segmentation\VNet\dataset"
# test_dir = os.path.join(save_dir, "val_data.h5")

# predictions = load_predictions(save_dir)
# x_test, y_test = load_dataset(test_dir)
# # x_test = np.squeeze(np.array(np.split(x_test, 8, axis=0)))
# # y_test = np.squeeze(np.array(np.split(y_test, 8, axis=0)))
# x_test = np.array(x_test)
# y_test = np.array(y_test)

# print_prediction(y_test, predictions,5, 50)


# -------------------------------------------------------------------------------below code is  final-----------------------------------
# import h5py
# import matplotlib.pyplot as plt
# import numpy as np
# import os


# # -------------------------
# # Load HDF5 dataset
# # -------------------------
# def load_dataset(path):
#     file = h5py.File(path, "r")

#     data = file["data"][:]        # CT scans
#     truth = file["truth"][:]      # Ground truth masks

#     # Ensure 4D (H,W,D) or 5D (N,H,W,D)
#     data = np.array(data)
#     truth = np.array(truth)

#     return data, truth


# # -------------------------
# # Load saved predictions
# # -------------------------
# def load_predictions(path):
#     pred_file = os.path.join(path, "predictions_vnet_at_128.h5")

#     print(f"Loading predictions from: {pred_file}")

#     file = h5py.File(pred_file, "r")
#     preds = file["predictions"][:]

#     print("Predictions shape:", preds.shape)
#     return preds


# # -------------------------
# # Visualization function
# # -------------------------
# def visualize_ct_ground_pred(ct, gt, pred, sample_index=0, slice_index=40):
#     """
#     Show:
#       - CT slice
#       - Ground Truth mask
#       - Predicted mask
#     """
#     ct_vol = ct[sample_index]
#     gt_vol = gt[sample_index]
#     pred_vol = pred[sample_index]

#     # squeeze channels if needed
#     ct_slice = np.squeeze(ct_vol[:, :, slice_index])
#     gt_slice = np.squeeze(gt_vol[:, :, slice_index])
#     pred_slice = np.squeeze(pred_vol[:, :, slice_index])

#     # binarize prediction (threshold)
#     pred_bin = (pred_slice > 0.5).astype(np.uint8)

#     plt.figure(figsize=(16, 7))

#     # 1) CT
#     plt.subplot(1, 2, 1)
#     plt.title("Ground Truth Mask")
#     plt.imshow(ct_slice, cmap="gray")
#     plt.axis("off")

#     # 2) Ground truth
#     plt.subplot(1, 2, 2)
#     plt.title("Predicted Mask")
#     plt.imshow(ct_slice, cmap="gray")
#     plt.imshow(gt_slice, cmap="jet", alpha=0.5)
#     plt.axis("off")

#     # 3) Prediction
#     # plt.subplot(1, 3, 3)
#     # plt.title("Apna Predicted Mask")
#     # plt.imshow(ct_slice, cmap="gray")
#     # plt.imshow(pred_bin, cmap="jet", alpha=0.5)
#     # plt.axis("off")

#     plt.tight_layout()
#     plt.show()


# # -------------------------
# # MAIN CODE
# # -------------------------
# save_dir = r"D:\SAM SEGMENTATION MASK\repo 6 bamf liver tumor segmentation\VNet\dataset"
# test_h5 = os.path.join(save_dir, "val_data.h5")

# # Load true data
# x_test, y_test = load_dataset(test_h5)

# # Load predicted masks
# predictions = load_predictions(save_dir)

# # Example visualization
# visualize_ct_ground_pred(
#     x_test,           # CT volumes
#     y_test,           # Ground truth masks
#     predictions,      # Model predictions
#     sample_index=1,   # which volume
#     slice_index=50    # which slice
# )
# ----------------------------------------------------------------------------------------------------
# from rich.console import Console
# from rich.panel import Panel
# from rich.prompt import IntPrompt
# from rich.spinner import Spinner
# from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
# from time import sleep
# import h5py
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# console = Console()


# # -------------------------
# # Load HDF5 dataset
# # -------------------------
# def load_dataset(path):
#     file = h5py.File(path, "r")
#     data = np.array(file["data"][:])
#     truth = np.array(file["truth"][:])
#     return data, truth


# # -------------------------
# # Load saved predictions
# # -------------------------
# def load_predictions(path):
#     pred_file = os.path.join(path, "predictions_vnet_at_128.h5")

#     console.print(Panel.fit(
#         f"[bold yellow]🔍 Loading Predictions[/bold yellow]\n{pred_file}",
#         border_style="bright_yellow"
#     ))

#     file = h5py.File(pred_file, "r")
#     preds = file["predictions"][:]

#     console.print(f"[bold green]✔ Predictions Loaded! Shape =[/bold green] {preds.shape}")
#     return preds


# # -------------------------
# # Visualization function
# # -------------------------
# def visualize_ct_ground_pred(ct, gt, pred, sample_index=0, slice_index=40):
#     ct_vol = ct[sample_index]
#     gt_vol = gt[sample_index]
#     pred_vol = pred[sample_index]

#     ct_slice = np.squeeze(ct_vol[:, :, slice_index])
#     gt_slice = np.squeeze(gt_vol[:, :, slice_index])
#     pred_slice = np.squeeze(pred_vol[:, :, slice_index])

#     pred_bin = (pred_slice > 0.5).astype(np.uint8)

#     plt.figure(figsize=(16, 7))

#     plt.subplot(1, 2, 1)
#     plt.title("Ground Truth Mask")
#     plt.imshow(ct_slice, cmap="gray")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.title("Predicted Mask")
#     plt.imshow(ct_slice, cmap="gray")
#     plt.imshow(gt_slice, cmap="jet", alpha=0.5)
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()


# # -------------------------
# # MAIN UI + LOGIC
# # -------------------------
# console.clear()

# # --- Welcome UI ---
# console.print(Panel.fit(
#     "[bold magenta]🩺 LIVER TUMOR SEGMENTATION – EVALUATION UI[/bold magenta]\n"
#     "[green]Developed for your Final Year Project[/green]\n\n"
#     "⭐ [yellow]Unique, Interactive Terminal Interface[/yellow]\n"
#     "⭐ [cyan]Enter sample index to view segmentation results[/cyan]",
#     border_style="magenta"
# ))

# # -------------------------
# # Load dataset
# # -------------------------
# save_dir = r"D:\SAM SEGMENTATION MASK\repo 6 bamf liver tumor segmentation\VNet\dataset"
# test_h5 = os.path.join(save_dir, "val_data.h5")

# console.print("\n[bold cyan]📂 Loading dataset...[/bold cyan]")

# with Progress(
#     SpinnerColumn(style="cyan"),
#     "[progress.description]{task.description}",
#     TimeElapsedColumn()
# ) as progress:
#     t = progress.add_task("Loading CT + Ground Truth...", total=None)
#     sleep(1)
#     x_test, y_test = load_dataset(test_h5)
#     progress.update(t, description="[green]Dataset Loaded Successfully![/green]")
#     sleep(0.5)

# # -------------------------
# # Load predictions
# # -------------------------
# predictions = load_predictions(save_dir)

# # -------------------------
# # Ask for sample index
# # -------------------------
# console.print("\n[bold yellow]🖼 Select sample to visualize[/bold yellow]\n")
# sample_index = IntPrompt.ask(
#     "[cyan]Enter sample index (0 to %d)[/cyan]" % (len(x_test)-1)
# )

# console.print(Panel.fit(
#     f"[bold blue]⏳ Processing Sample {sample_index}...[/bold blue]\n"
#     "Model is analyzing tumor regions...",
#     border_style="bright_blue"
# ))

# # Fake “DETECTING” animation
# with Progress(
#     SpinnerColumn(style="yellow"),
#     "[progress.description]{task.description}",
#     TimeElapsedColumn()
# ) as progress:
#     t = progress.add_task("Detecting tumors...", total=None)
#     sleep(2)
#     progress.update(t, description="[green]Detection complete![/green]")
#     sleep(0.5)

# console.print(Panel.fit(
#     f"[bold green]✔ Visualization Ready![/bold green]\nDisplaying CT, Ground Truth & Predicted Mask",
#     border_style="green"
# ))

# # -------------------------
# # Show output
# # -------------------------
# visualize_ct_ground_pred(
#     x_test,
#     y_test,
#     predictions,
#     sample_index=sample_index,
#     slice_index=50
# )





















# ----------------------------------------------------------------------------------------
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox


# -------------------------
# Load HDF5 dataset
# -------------------------
def load_dataset(path):
    file = h5py.File(path, "r")
    data = np.array(file["data"][:])
    truth = np.array(file["truth"][:])
    return data, truth


# -------------------------
# Load predictions
# -------------------------
def load_predictions(path):
    pred_file = os.path.join(path, "predictions_vnet_at_128.h5")
    file = h5py.File(pred_file, "r")
    preds = file["predictions"][:]
    return preds


# -------------------------
# Visualization
# -------------------------
def visualize_ct_ground_pred(ct, gt, pred, sample_index=0, slice_index=40):
    ct_vol = ct[sample_index]
    gt_vol = gt[sample_index]
    pred_vol = pred[sample_index]

    ct_slice = np.squeeze(ct_vol[:, :, slice_index])
    gt_slice = np.squeeze(gt_vol[:, :, slice_index])
    pred_slice = np.squeeze(pred_vol[:, :, slice_index])

    pred_bin = (pred_slice > 0.5).astype(np.uint8)

    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth Mask")
    plt.imshow(ct_slice, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(ct_slice, cmap="gray")
    plt.imshow(gt_slice, cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# Load data first (outside UI)
# -------------------------
save_dir = r"D:\SAM SEGMENTATION MASK\repo 6 bamf liver tumor segmentation\VNet\dataset"
test_h5 = os.path.join(save_dir, "val_data.h5")

x_test, y_test = load_dataset(test_h5)
predictions = load_predictions(save_dir)


# ===============================================================
#                       TKINTER POPUP UI
# ===============================================================
# def start_ui():
#     root = tk.Tk()
#     root.title("Liver Tumor Segmentation - Evaluation UI")
#     root.geometry("450x350")
#     root.configure(bg="#1e1e1e")

#     title = tk.Label(root, text="🩺 Liver Tumor Segmentation", fg="white", bg="#1e1e1e",
#                      font=("Arial", 50, "bold"))
#     title.pack(pady=15)

#     subtitle = tk.Label(root, text="Enter Sample Index to Visualize", fg="#00e6e6", bg="#1e1e1e",
#                         font=("Arial", 30))
#     subtitle.pack()

#     index_var = tk.StringVar()

#     sample_entry = ttk.Entry(
#         root,
#         textvariable=index_var,
#         width=35,            # wider
#         font=("Arial", 14),  # bigger text
#     )
#     sample_entry.pack(pady=25, ipady=8)     

#     # Label for animations
#     status_label = tk.Label(root, text="", fg="yellow", bg="#1e1e1e", font=("Arial", 20))
#     status_label.pack(pady=10)

#     # Function to animate detecting...
#     def detecting_animation():
#         for i in range(6):
#             status_label.config(text="🔍 Detecting" + "." * (i % 4))
#             time.sleep(1.4)

#         status_label.config(text="✔ Detection Complete!", fg="lightgreen")

#     # Button function
#     def detect():
#         try:
#             idx = int(index_var.get())
#             if idx < 0 or idx >= len(x_test):
#                 messagebox.showerror("Error", f"Index must be between 0 and {len(x_test)-1}")
#                 return
#         except:
#             messagebox.showerror("Error", "Please enter a valid number")
#             return

#         status_label.config(text="🔍 Detecting...", fg="yellow")

#         # Run animation in thread
#         thread = threading.Thread(target=detecting_animation)
#         thread.start()

#     def visualize():
#         try:
#             idx = int(index_var.get())
#             visualize_ct_ground_pred(x_test, y_test, predictions, sample_index=idx, slice_index=50)
#         except:
#             messagebox.showerror("Error", "Enter a valid index first")

#     def reset():
#         index_var.set("")
#         status_label.config(text="", fg="yellow")

#     # Buttons
#     button_style = {
#     "width": 20,
#     "padding": 10
# }

#     btn_detect = ttk.Button(root, text="▶ Start Detection", command=detect)
#     btn_detect.pack(pady=10, ipadx=15, ipady=8)

#     btn_show = ttk.Button(root, text="📊 Show Output", command=visualize)
#     btn_show.pack(pady=10, ipadx=15, ipady=8)

#     btn_reset = ttk.Button(root, text="🔄 Try Another Sample", command=reset)
#     btn_reset.pack(pady=15, ipadx=15, ipady=8)

#     root.mainloop()


# # Run UI
# start_ui()
# -----------------------------------------------------------------------------------------------------------------------------------
def start_ui():
    root = tk.Tk()
    root.title("Liver Tumor Segmentation - Evaluation UI")
    root.geometry("600x500")       # bigger window for large fonts
    root.configure(bg="#1e1e1e")

    # ---------------- Title ----------------
    title = tk.Label(root,
                     text="🩺 Liver Tumor Segmentation",
                     fg="white",
                     bg="#1e1e1e",
                     font=("Arial", 32, "bold"))
    title.pack(pady=10)

    subtitle = tk.Label(root,
                        text="Enter Sample Index",
                        fg="#00e6e6",
                        bg="#1e1e1e",
                        font=("Arial", 22))
    subtitle.pack()

    # ---------------- Input Box + Reset in one Frame ----------------
    input_frame = tk.Frame(root, bg="#1e1e1e")
    input_frame.pack(pady=10)

    index_var = tk.StringVar()

    sample_entry = ttk.Entry(
        input_frame,
        textvariable=index_var,
        width=35,
        font=("Arial", 14)
    )
    sample_entry.grid(row=0, column=0, padx=10, ipady=10)

    # Try Another Sample button (moved next to input box)
    btn_reset = ttk.Button(input_frame,
                           text="🔄 Reset",
                           command=lambda: [index_var.set(""), status_label.config(text="")])
    btn_reset.grid(row=0, column=1, padx=10, ipadx=10, ipady=8)

    # ---------------- Status Label ----------------
    status_label = tk.Label(root,
                            text="",
                            fg="yellow",
                            bg="#1e1e1e",
                            font=("Arial", 20))
    status_label.pack(pady=20)

    # ---------------- Detect Animation ----------------
    def detecting_animation(idx):
        for i in range(6):
            status_label.config(text="🔍 Detecting" + "." * (i % 4))
            time.sleep(0.6)

        # Switch to main thread → safe to update UI + call plt.show()
        root.after(0, lambda: finish_visualization(idx))
    
    
    def finish_visualization(idx):
        status_label.config(text="✔ Detection Complete!", fg="lightgreen")

        # Safe call: Matplotlib on main thread
        visualize_ct_ground_pred(
            x_test, y_test, predictions,
            sample_index=idx,
            slice_index=50
    )
        

    # ---------------- Detect Function ----------------
    def detect():
        try:
            idx = int(index_var.get())
            if idx < 0 or idx >= len(x_test):
                messagebox.showerror("Error", f"Index must be between 0 and {len(x_test)-1}")
                return
        except:
            messagebox.showerror("Error", "Please enter a valid number")
            return

        status_label.config(text="🔍 Detecting...", fg="yellow")

        # Run detection animation + visualize in thread
        thread = threading.Thread(target=detecting_animation, args=(idx,))
        thread.start()

    # ---------------- Buttons ----------------
    btn_detect = ttk.Button(root,
                            text="▶ Start Detection",
                            command=detect)
    btn_detect.pack(pady=25, ipadx=20, ipady=12)

    root.mainloop()


# Run UI
start_ui()
