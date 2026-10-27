from rfdetr import RFDETRMedium
 
if __name__ == "__main__":   # required on Windows (DataLoader workers)
    model = RFDETRMedium()
    model.train(
        dataset_dir=r"D:\Dre\rf-detr\dataset",
        epochs=50,
        batch_size=8,
        grad_accum_steps=2,    # effective batch = 8 x 2 = 16
        lr=1e-4,
        output_dir=r"D:\Dre\rf-detr\output",
    )