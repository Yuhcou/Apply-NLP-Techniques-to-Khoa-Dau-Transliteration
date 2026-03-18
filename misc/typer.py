import tkinter as tk
from quoc_ngu_to_khoa_dau.rule_based import encode_custom

def on_text_change(event=None):
    encoded = encode_custom(entry.get())
    output_label.config(text=encoded)

root = tk.Tk()
root.title("PUA Font Encoder")

entry = tk.Entry(root, font=("Segoe UI", 14), width=40)
entry.pack(padx=20, pady=10)
entry.bind("<KeyRelease>", on_text_change)

output_label = tk.Label(
    root,
    text="",
    font=("Khoa Dau", 48),  # font PUA của bạn
    bg="white"
)
output_label.pack(padx=20, pady=20, fill="both")

on_text_change()
root.mainloop()