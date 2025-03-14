# Authors: Hugo Richard, Badr Tajini
# License: BSD 3 clause

import sys
import os
import glob
import nibabel as nib
import numpy as np
from joblib import load, dump
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like
from scipy.ndimage import zoom

# Add the main package path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from condica.main import condica
from condica.utils import fetch_difumo

# Function to resize NIfTI images to a common shape
def resize_nifti(data, target_shape):
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
    return zoom(data, zoom_factors, order=1)  # Linear interpolation

def main():
    print("🔥 Script started!")  # Debug print
    try:
        # 🔹 Set path to your NIfTI images
        anat_path = r"C:\Users\Shiv Desai\Desktop\anat_images\anat"

        # 🔹 Check if directory exists
        if not os.path.exists(anat_path):
            raise FileNotFoundError(f"❌ Directory not found: {anat_path}")

        print(f"📂 Checking directory: {anat_path}")

        # 🔹 Load all NIfTI files
        nifti_files = glob.glob(os.path.join(anat_path, "*.nii")) + glob.glob(os.path.join(anat_path, "*.nii.gz"))

        # 🔹 Check if NIfTI files exist
        if not nifti_files:
            raise FileNotFoundError(f"❌ No NIfTI files found in: {anat_path}")

        print(f"✅ Found {len(nifti_files)} NIfTI files.")
    
        # 🔹 Convert NIfTI files to NumPy arrays
        TARGET_SHAPE = (208, 300, 320)  # Modify based on your needs

        data = []
        for file in nifti_files:
            print(f"⏳ Loading {file}...")
            try:
                img = nib.load(file)
                img_data = img.get_fdata()

                # Resize to TARGET_SHAPE if necessary
                if img_data.shape != TARGET_SHAPE:
                    print(f"⚠️ Resizing {file} from {img_data.shape} to {TARGET_SHAPE}")
                    img_data = resize_nifti(img_data, TARGET_SHAPE)

                data.append(img_data)
                print(f"✅ Loaded {file}, final shape: {img_data.shape}")

            except Exception as e:
                raise RuntimeError(f"❌ Error loading NIfTI file: {file}\n{e}")

        X = np.array(data)  # Now X will have a consistent shape

        # 🔹 Check if data was loaded correctly
        if X.ndim < 3:
            raise ValueError("❌ Loaded NIfTI data has incorrect dimensions. Expected 3D+ images.")

        print(f"✅ NIfTI data loaded successfully. Shape: {X.shape}")

        # 🔹 Check for brain mask file
        mask_path = "../data/hcp_mask.nii.gz"
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"❌ Brain mask not found: {mask_path}")
        print(f"✅ Using brain mask: {mask_path}")

        # 🔹 Convert NumPy array to a 4D NIfTI image before applying the brain mask
        print("⏳ Converting data to 4D NIfTI image before masking...")
        affine = np.eye(4)  # Assuming default affine, modify if needed
        X_nifti = new_img_like(mask_path, X, affine)

        print(f"✅ Data converted to 4D NIfTI image. Shape: {X_nifti.shape}")

        # Apply the brain mask
        try:
            masker = NiftiMasker(mask_img=mask_path, verbose=1)
            X_masked = masker.fit_transform(X_nifti)  # Extract only brain voxels
            print(f"✅ Brain mask applied successfully. New shape: {X_masked.shape}")
        except Exception as e:
            raise RuntimeError(f"❌ Error applying brain mask: {e}")

        # 🔹 Fetch Difumo atlas
        print("⏳ Fetching Difumo atlas...")
        try:
            mask = fetch_difumo(dimension=1024, data_dir="../data/").maps
            components = (
                NiftiMasker(mask_img=mask_path, verbose=1)
                .fit()
                .transform(mask)
            )
            dump(components, "../data/difumo_atlases/1024/components.pt")
            C = load("../data/difumo_atlases/1024/components.pt")
            print(f"✅ Difumo atlas loaded. Shape: {C.shape}")
        except Exception as e:
            raise RuntimeError(f"❌ Error fetching Difumo atlas: {e}")

        # 🔹 Ensure the masked data and Difumo atlas match in shape
        if X_masked.shape[1] != C.shape[0]:
            print(f"⚠️ Shape mismatch: X_masked has {X_masked.shape[1]} voxels, but Difumo expects {C.shape[0]}. Transposing Difumo atlas.")
            C = C.T  # ✅ Fix the shape mismatch by transposing

        # 🔹 Reduce the data using the atlas
        print("⏳ Reducing data using Difumo atlas...")
        try:
            X_reduced = X_masked @ C  # Matrix multiplication
            print(f"✅ Data reduced. New shape: {X_reduced.shape}")
        except Exception as e:
            raise RuntimeError(f"❌ Error applying atlas transformation: {e}")
        
         # 🔹 Load mixing matrix
        A_rest_path = "../data/A_rest.npy"
        if not os.path.exists(A_rest_path):
            raise FileNotFoundError(f"❌ Missing required file: {A_rest_path}")

        print("⏳ Loading mixing matrix...")
        try:
            A = np.load(A_rest_path)
            print("✅ Mixing matrix loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading A_rest.npy: {e}")
        
        # 🔹 Define Y (labels)
       # Create dummy labels for 23 samples (modify if needed)
        num_samples = X_reduced.shape[0]
        Y = np.array([i % 2 for i in range(num_samples)])  # Alternates 0,1,0,1...

        print(f"✅ Labels created. Shape: {Y.shape}, Values: {Y}")

        # 🔹 Machine Learning Pipeline
        clf = LinearDiscriminantAnalysis()
        cv = ShuffleSplit(random_state=0, train_size=0.8, n_splits=20)

        scores_noaug = []
        scores_withaug = []

        print("⏳ Running classification with and without augmentation...")
        for train, test in cv.split(X_reduced):
            print("🔹 Running train-test split...")
            X_train, X_test = X_reduced[train], X_reduced[test]
            Y_train, Y_test = Y[train], Y[test]

        # 🔹 Generate synthetic fMRI data using CondICA
        print("⏳ Generating synthetic data using CondICA...")
        try:
            X_fakes, Y_fakes = condica(A, X_train, Y_train, len(X_train), n_quantiles=len(X_train))
            print("✅ Synthetic data generated.")
        except Exception as e:
            raise RuntimeError(f"❌ Error running CondICA augmentation: {e}")

        # 🔹 Train without augmentation
        print("⏳ Training classifier WITHOUT augmentation...")
        try:
            scores_noaug.append(clf.fit(X_train, Y_train).score(X_test, Y_test))
            print("✅ Classifier trained WITHOUT augmentation.")
        except Exception as e:
            raise RuntimeError(f"❌ Error training classifier without augmentation: {e}")
        
          # 🔹 Train with augmentation
        print("⏳ Training classifier WITH augmentation...")
        try:
            scores_withaug.append(
                clf.fit(
                    np.concatenate([X_train, X_fakes]),
                    np.concatenate([Y_train, Y_fakes]),
                ).score(X_test, Y_test)
            )
            print("✅ Classifier trained WITH augmentation.")
        except Exception as e:
            raise RuntimeError(f"❌ Error training classifier with augmentation: {e}")

        # 🔹 Print final results
        print("✅ Classification complete.")
        print(f"📊 Mean accuracy WITHOUT augmentation: {np.mean(scores_noaug):.3f}")
        print(f"📊 Mean accuracy WITH augmentation: {np.mean(scores_withaug):.3f}")


    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)  # Stop execution on failure

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    try:
        main()
    except Exception as e:
        print(f"❌ ERROR in main(): {e}")
