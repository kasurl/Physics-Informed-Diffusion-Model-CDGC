import numpy as np
import os
from PIL import Image

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        classAcc = np.diag(self.confusionMatrix) / np.maximum(self.confusionMatrix.sum(axis=1), 1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.maximum(np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix), 1)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        """Generate confusion matrix for 512×512 grayscale images"""
        # Convert to single channel
        if len(imgPredict.shape) > 2:
            imgPredict = imgPredict[:, :, 0]
        if len(imgLabel.shape) > 2:
            imgLabel = imgLabel[:, :, 0]

        # Binarize images
        imgPredict = (imgPredict > 0).astype(np.int64)
        imgLabel = (imgLabel > 0).astype(np.int64)

        # Filter valid labels
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label_indices = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label_indices, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def evaluate_single_image(predict_path, label_path):
    """Evaluate single 512×512 grayscale image"""
    imgPredict = Image.open(predict_path).convert('L')
    imgPredict = np.array(imgPredict)
    imgLabel = Image.open(label_path).convert('L')
    imgLabel = np.array(imgLabel)

    metric = SegmentationMetric(numClass=2)
    metric.addBatch(imgPredict, imgLabel)

    acc = metric.pixelAccuracy()
    macc = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    fwIoU = metric.Frequency_Weighted_Intersection_over_Union()

    print(f"Single image evaluation:")
    print(f"PA: {acc:.4f}, MPA: {macc:.4f}, mIoU: {mIoU:.4f}, FW-IoU: {fwIoU:.4f}")
    return acc, macc, mIoU, fwIoU

def evaluate_batch(pre_dir, label_dir):
    """Batch evaluate 512×512 grayscale images"""
    assert os.path.exists(pre_dir), f"Prediction dir missing: {pre_dir}"
    assert os.path.exists(label_dir), f"Label dir missing: {label_dir}"

    pre_filenames = sorted([f for f in os.listdir(pre_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
    label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
    assert len(pre_filenames) == len(label_filenames), "Image count mismatch"

    acc_list, macc_list, mIoU_list, fwIoU_list = [], [], [], []
    global_metric = SegmentationMetric(numClass=2)

    for pre_name, label_name in zip(pre_filenames, label_filenames):
        pre_path = os.path.join(pre_dir, pre_name)
        label_path = os.path.join(label_dir, label_name)

        imgPredict = Image.open(pre_path).convert('L')
        imgPredict = np.array(imgPredict)
        imgLabel = Image.open(label_path).convert('L')
        imgLabel = np.array(imgLabel)

        single_metric = SegmentationMetric(numClass=2)
        single_metric.addBatch(imgPredict, imgLabel)
        acc_list.append(single_metric.pixelAccuracy())
        macc_list.append(single_metric.meanPixelAccuracy())
        mIoU_list.append(single_metric.meanIntersectionOverUnion())
        fwIoU_list.append(single_metric.Frequency_Weighted_Intersection_over_Union())

        global_metric.addBatch(imgPredict, imgLabel)
        print(f"Processed: {pre_name} | PA: {acc_list[-1]:.4f}, mIoU: {mIoU_list[-1]:.4f}")

    # Calculate metrics
    avg_acc, avg_macc, avg_mIoU, avg_fwIoU = np.mean(acc_list), np.mean(macc_list), np.mean(mIoU_list), np.mean(fwIoU_list)
    global_acc, global_macc, global_mIoU, global_fwIoU = global_metric.pixelAccuracy(), global_metric.meanPixelAccuracy(), global_metric.meanIntersectionOverUnion(), global_metric.Frequency_Weighted_Intersection_over_Union()

    print("\n" + "="*50)
    print("Batch evaluation results (512×512 grayscale, binary)")
    print("="*50)
    print(f"Single average - PA: {avg_acc:.4f}, MPA: {avg_macc:.4f}, mIoU: {avg_mIoU:.4f}, FW-IoU: {avg_fwIoU:.4f}")
    print(f"Global matrix - PA: {global_acc:.4f}, MPA: {global_macc:.4f}, mIoU: {global_mIoU:.4f}, FW-IoU: {global_fwIoU:.4f}")
    print("="*50)

    return (acc_list, macc_list, mIoU_list, fwIoU_list), (global_acc, global_macc, global_mIoU, global_fwIoU)

if __name__ == '__main__':
    PRE_DIR = '../path_to_predictions'
    LABEL_DIR = '../path_to_labels'
    _, global_metrics = evaluate_batch(PRE_DIR, LABEL_DIR)