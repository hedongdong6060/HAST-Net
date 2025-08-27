#
# # # print(f"Output predictions saved to {save_path}")
# from torch.utils.data import Dataset, DataLoader
# import pytorch_lightning as pl
# from dataloader.New_PolypVideoDataset import SlidingWindowClipSampler
# from argparse import ArgumentParser
# import numpy as np
# from util.util import * # Assumes build_data, build_model are here
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.callbacks import LearningRateFinder
# import yaml
# import cv2 # Ensure导入cv2
# import os
# import torch
# import warnings
# import traceback # For detailed error printing
#
# warnings.filterwarnings("ignore")
#
# parser = ArgumentParser()
# parser.add_argument(
#     "--cfg",
#     type=str,
#     default="configs/New_PolypVideoDataset.yaml", # Ensure this path is correct
#     help="Configuration file to use",
# )
# parser.add_argument(
#     "--output_pred",
#     type=str,
#     default="./predictions",
#     help="Path to save output predictions",
# )
# parser.add_argument(
#     "--checkpoint",
#     type=str,
#     # !!! CRITICAL: ENSURE THIS PATH IS CORRECT AND POINTS TO A WaveletNetPlus CHECKPOINT !!!
#     default="checkpoints/xxx/xx/New_PolypVideoDataset_15/New_PolypVideoDataset_15.ckpt",
#     help="Path to the FULLY TRAINED WaveletNetPlus model checkpoint.",
# )
# parser.add_argument(
#     "--overlay",
#     action='store_true', # Use action='store_true' for boolean flags
#     help="Generate the overlay image (original image blended with the mask).",
# )
# # Optional: Add a flag to disable saving frequency maps for faster testing
# parser.add_argument(
#     "--no_freq_maps",
#     action='store_true',
#     help="Disable saving of frequency maps.",
# )
#
# args = parser.parse_args()
#
# # --- 1. Load Configuration ---
# print(f"Loading configuration from: {args.cfg}")
# if not os.path.exists(args.cfg):
#     print(f"Error: Configuration file not found at {args.cfg}")
#     exit()
# with open(args.cfg) as f:
#     cfg = yaml.load(f, Loader=yaml.SafeLoader)
# print("Configuration loaded successfully.")
#
# # --- 2. Build Dataset and DataLoader ---
# print("Building validation dataset...")
# try:
#     # Assuming build_data returns (train_set, val_set)
#     _, val_set = build_data(cfg) # val_set is New_PolypVideoDataset instance
#     if not val_set or len(val_set) == 0:
#          print("Error: Validation dataset is empty or failed to build.")
#          exit()
#     print(f"Validation dataset built. Number of videos: {len(val_set)}")
#     args.class_list = val_set.CLASSES # Needed for build_model potentially
# except Exception as e:
#     print(f"Error building dataset: {e}")
#     traceback.print_exc()
#     exit()
#
# save_path = os.path.join(args.output_pred, cfg["DATASET"]["dataset"])
# print(f"Predictions will be saved under: {save_path}")
#
# # --- Create DataLoader ---
# clip_len_pred = cfg["DATASET"].get("clip_len_pred", 8) # Use same clip_len as val_set init or specify
# stride_pred = cfg["DATASET"].get("stride_pred", clip_len_pred) # Default stride to clip_len if not specified
# print(f"Using clip_len={clip_len_pred}, stride={stride_pred} for prediction sampling.")
#
# sampler_val = SlidingWindowClipSampler(
#     dataset=val_set,
#     clip_len=clip_len_pred,
#     stride=stride_pred,
#     shuffle=False,
#     drop_last=False
# )
#
# # Check sampler length
# try:
#     sampler_len = len(sampler_val)
#     print(f"Sampler created. Expected number of clips: {sampler_len}")
#     if sampler_len == 0:
#         print("Warning: Sampler generated 0 clips. Check dataset/sampler parameters.")
#         # Decide whether to exit or continue (maybe dataset is small?)
#         # exit()
# except Exception as e:
#     print(f"Warning: Could not get sampler length: {e}")
#
#
# data_loader_val = DataLoader(
#     val_set,
#     sampler=sampler_val,
#     batch_size=1, # CRITICAL: Must be 1 when using sampler for clips
#     num_workers=cfg["TRAIN"].get("num_workers", 0), # Get num_workers safely
#     pin_memory=True
# )
# print(f"DataLoader created. Number of batches (clips): {len(data_loader_val)}")
# if len(data_loader_val) == 0:
#     print("Error: DataLoader is empty. Cannot proceed with prediction.")
#     exit()
#
# # --- 3. Build Model ---
# print("Building model...")
# try:
#     # Pass args to build_model if it needs class_list etc.
#     model = build_model(args, cfg)
#     print("Model built successfully.")
# except Exception as e:
#     print(f"Error building model: {e}")
#     traceback.print_exc()
#     exit()
#
# # --- 4. Setup Trainer ---
# print("Setting up PyTorch Lightning Trainer...")
# trainer = pl.Trainer(
#     devices=1,
#     accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     logger=False, # No logging needed for prediction
#     enable_model_summary=False,
#     enable_progress_bar=True # Show progress bar during predict
# )
# print(f"Trainer ready. Using accelerator: {trainer.accelerator}")
#
# # --- 5. Perform Prediction ---
# print(f"\n--- Starting prediction using checkpoint: {args.checkpoint} ---")
# if not os.path.exists(args.checkpoint):
#     print(f"Error: Checkpoint file not found at {args.checkpoint}")
#     exit()
#
# predictions = None
# try:
#     predictions = trainer.predict(
#         model, dataloaders=data_loader_val, ckpt_path=args.checkpoint
#     )
# except Exception as e:
#     print(f"\n--- Error during trainer.predict ---")
#     print(f"Exception type: {type(e)}")
#     print(f"Error message: {e}")
#     print("Traceback:")
#     traceback.print_exc()
#     print("--- End of Error ---")
#     # Optionally try to proceed if predictions might have partial results
#     # but it's safer to exit if predict itself failed.
#     predictions = [] # Ensure predictions is a list even on error
#
# # --- CRITICAL CHECK ---
# if predictions is None:
#      print("\nError: trainer.predict returned None. Prediction failed.")
#      exit()
# elif not isinstance(predictions, list):
#      print(f"\nError: trainer.predict returned type {type(predictions)}, expected list.")
#      exit()
# elif len(predictions) == 0:
#      print("\nWarning: trainer.predict returned an empty list []. No predictions were generated.")
#      print("Possible reasons: Empty DataLoader, error inside predict_step, OOM.")
#      # Exit or continue depending on whether an empty result is acceptable
#      exit()
# else:
#     print(f"\nPrediction finished. Got {len(predictions)} prediction items (clips).")
#     # Optional: Print structure of the first item for debugging
#     # print("Structure of the first prediction item:")
#     # first_item = predictions[0]
#     # if isinstance(first_item, dict):
#     #     for key, value in first_item.items():
#     #         if isinstance(value, torch.Tensor):
#     #             print(f"  Key: '{key}', Type: Tensor, Shape: {value.shape}, Dtype: {value.dtype}")
#     #         elif isinstance(value, list):
#     #             print(f"  Key: '{key}', Type: List, Length: {len(value)}")
#     #         else:
#     #             print(f"  Key: '{key}', Type: {type(value)}")
#     # else:
#     #      print(f"  First item is not a dict, type: {type(first_item)}")
#
# # --- 6. Process and Save Predictions ---
# color_map = {
#     0: (0, 0, 0),       # background Black (RGB)
#     1: (0, 255, 0),     # Polyp Green (RGB)
#     2: (0, 0, 255),     # instrument Blue (RGB)
# }
#
#
#
#
#
#
#
#
#
# print("\n--- Starting saving loop... ---")
# saved_count = 0
# skipped_count = 0
# for item_idx, item in enumerate(predictions):
#     # --- Item Validation ---
#     if not isinstance(item, dict) or not all(k in item for k in ["preds", "img_paths", "low_freqs", "high_freqs"]):
#         print(f"Warning [Clip {item_idx}]: Unexpected item format. Skipping. Keys: {item.keys() if isinstance(item, dict) else 'Not a dict'}")
#         skipped_count += 1 # Assuming all frames in this item are skipped
#         continue
#
#     if not isinstance(item["preds"], torch.Tensor):
#          print(f"Warning [Clip {item_idx}]: 'preds' is not a Tensor. Skipping.")
#          skipped_count += 1
#          continue
#     if not isinstance(item["img_paths"], list):
#          print(f"Warning [Clip {item_idx}]: 'img_paths' is not a List. Skipping.")
#          skipped_count += 1
#          continue
#     if not isinstance(item["low_freqs"], torch.Tensor) or not isinstance(item["high_freqs"], torch.Tensor):
#          print(f"Warning [Clip {item_idx}]: 'low_freqs' or 'high_freqs' is not a Tensor. Skipping.")
#          skipped_count += 1
#          continue
#
#     try:
#         num_frames = item["preds"].shape[0]
#         all_preds = item["preds"].cpu().numpy().astype(np.uint8) # (T, H, W)
#         all_img_paths = item["img_paths"]                 # List of T paths
#         all_low_freqs = item["low_freqs"].cpu().numpy()   # (T, C, H_in, W_in)
#         all_high_freqs = item["high_freqs"].cpu().numpy() # (T, C, H_in, W_in)
#     except Exception as e:
#         print(f"Error unpacking data from item {item_idx}: {e}. Skipping clip.")
#         skipped_count += item.get("preds", torch.empty(0)).shape[0] # Estimate skipped frames
#         continue
#
#     # --- Data Consistency Check ---
#     if not (len(all_img_paths) == num_frames and all_low_freqs.shape[0] == num_frames and all_high_freqs.shape[0] == num_frames):
#         print(f"Warning [Clip {item_idx}]: Data length mismatch. Frames: {num_frames}, Paths: {len(all_img_paths)}, LowFreq: {all_low_freqs.shape[0]}, HighFreq: {all_high_freqs.shape[0]}. Skipping clip.")
#         skipped_count += num_frames
#         continue
#
#     print(f"Processing clip {item_idx+1}/{len(predictions)}, {num_frames} frames...")
#
#     # --- Loop Through Frames in the Clip ---
#     for i in range(num_frames):
#         frame_saved = False # Flag to track if anything was saved for this frame
#         try:
#             pred = all_preds[i]           # (H, W) uint8 mask
#             img_path = all_img_paths[i]   # string path
#             low_freq = all_low_freqs[i]   # (C, H_in, W_in) float array
#             high_freq = all_high_freqs[i] # (C, H_in, W_in) float array
#
#             # --- Path Manipulation (Robust Approach) ---
#             if not img_path or not isinstance(img_path, str):
#                 print(f"Warning [Clip {item_idx}, Frame {i}]: Invalid image path '{img_path}'. Skipping frame.")
#                 skipped_count += 1
#                 continue
#
#             norm_path = os.path.normpath(img_path)
#             parts = norm_path.split(os.sep)
#             if len(parts) < 2:
#                 print(f"Warning [Clip {item_idx}, Frame {i}]: Cannot determine parent folder from path '{img_path}'. Saving to 'unknown_subfolder'.")
#                 subfolder = "unknown_subfolder"
#                 file_name = parts[-1] if parts else "unknown_frame"
#             else:
#                 # Use parent directory name as subfolder (more robust than relying on 'test')
#                 subfolder = parts[-2]
#                 file_name = parts[-1]
#
#             base_name, _ = os.path.splitext(file_name)
#
#             # Construct save directory
#             save_dir = os.path.join(save_path, subfolder)
#             os.makedirs(save_dir, exist_ok=True)
#
#             # --- Mask Saving Logic ---
#             pred_h, pred_w = pred.shape
#             # Create BGR color mask directly (Assuming New_PolypVideoDataset)
#             color_mask_bgr = np.zeros((pred_h, pred_w, 3), dtype=np.uint8)
#             for class_id, color_rgb in color_map.items():
#                 color_bgr = tuple(reversed(color_rgb)) # RGB to BGR for OpenCV
#                 color_mask_bgr[pred == class_id] = color_bgr
#
#             # Save the non-overlaid prediction mask
#             not_overlay_dir = os.path.join(save_dir, "not_overlay")
#             os.makedirs(not_overlay_dir, exist_ok=True)
#             not_overlay_filename = os.path.join(not_overlay_dir, f"{base_name}_pred.png")
#             try:
#                 cv2.imwrite(not_overlay_filename, color_mask_bgr)
#                 frame_saved = True # Mark as saved
#             except Exception as e:
#                 print(f"Error saving non-overlay mask {not_overlay_filename}: {e}")
#                 # Continue to next steps for this frame if needed, or skip frame
#                 # continue # Uncomment to skip frame entirely on mask save error
#
#             # --- Overlay Logic ---
#             if args.overlay:
#                 orig_img = cv2.imread(img_path)
#                 if orig_img is None:
#                     print(f"Warning [Clip {item_idx}, Frame {i}]: Failed to read image: {img_path}. Skipping overlay & freq maps.")
#                     # If original image fails, skip overlay and potentially freq maps
#                     # continue # Uncomment to skip rest of processing for this frame
#                 else:
#                     try:
#                         # Resize original image to match prediction mask dimensions (W, H for cv2.resize)
#                         orig_img_resized = cv2.resize(orig_img, (pred_w, pred_h))
#
#                         # Blend original image and prediction mask
#                         if color_mask_bgr.shape != orig_img_resized.shape:
#                              print(f"Warning [Clip {item_idx}, Frame {i}]: Shape mismatch mask {color_mask_bgr.shape} vs resized img {orig_img_resized.shape} for {img_path}. Skipping overlay.")
#                         else:
#                             overlay = cv2.addWeighted(orig_img_resized, 0.6, color_mask_bgr, 0.4, 0)
#
#                             # Save overlay image
#                             overlay_dir = os.path.join(save_dir, "overlay")
#                             os.makedirs(overlay_dir, exist_ok=True)
#                             overlay_filename = os.path.join(overlay_dir, f"{base_name}_overlay.png")
#                             try:
#                                 cv2.imwrite(overlay_filename, overlay)
#                                 frame_saved = True
#                             except Exception as e:
#                                 print(f"Error saving overlay image {overlay_filename}: {e}")
#
#                     except Exception as e:
#                         print(f"Error processing overlay for {img_path}: {e}")
#
#             # --- Frequency Map Saving ---
#             if not args.no_freq_maps:
#                 # De-normalize and save Low Frequency
#                 try:
#                     # Safely get mean/std, assuming they are lists/tuples
#                     mean_low = cfg["DATASET"]["Low_mean"]
#                     std_low = cfg["DATASET"]["Low_std"]
#                     if isinstance(mean_low, (list, tuple)) and len(mean_low) > 0: mean_low = mean_low[0]
#                     if isinstance(std_low, (list, tuple)) and len(std_low) > 0: std_low = std_low[0]
#
#                     # low_freq shape is (C, H_in, W_in), assuming C=1
#                     denorm_low_freq = (low_freq.squeeze(0) * std_low + mean_low) * 255.0 # Squeeze Channel dim
#                     low_freq_resized = cv2.resize(denorm_low_freq, (pred_w, pred_h))
#                     low_freq_resized = np.clip(low_freq_resized, 0, 255).astype(np.uint8)
#
#                     low_freq_dir = os.path.join(save_dir, "low_freq")
#                     os.makedirs(low_freq_dir, exist_ok=True)
#                     low_freq_filename = os.path.join(low_freq_dir, f"{base_name}_low.png")
#                     cv2.imwrite(low_freq_filename, low_freq_resized)
#                     frame_saved = True
#
#                 except KeyError as e:
#                      print(f"Warning [Clip {item_idx}, Frame {i}]: Missing Low Freq norm key in config: {e}. Skipping low freq.")
#                 except Exception as e:
#                      print(f"Error processing/saving low frequency map for {img_path}: {e}")
#
#                 # De-normalize and save High Frequency
#                 try:
#                     # Safely get mean/std
#                     mean_high = cfg["DATASET"]["High_mean"]
#                     std_high = cfg["DATASET"]["High_std"]
#                     if isinstance(mean_high, (list, tuple)) and len(mean_high) > 0: mean_high = mean_high[0]
#                     if isinstance(std_high, (list, tuple)) and len(std_high) > 0: std_high = std_high[0]
#
#                     # high_freq shape is (C, H_in, W_in), assuming C=1
#                     denorm_high_freq = (high_freq.squeeze(0) * std_high + mean_high) * 255.0 # Squeeze Channel dim
#                     high_freq_resized = cv2.resize(denorm_high_freq, (pred_w, pred_h))
#                     high_freq_resized = np.clip(high_freq_resized, 0, 255).astype(np.uint8)
#
#                     high_freq_dir = os.path.join(save_dir, "high_freq")
#                     os.makedirs(high_freq_dir, exist_ok=True)
#                     high_freq_filename = os.path.join(high_freq_dir, f"{base_name}_high.png")
#                     cv2.imwrite(high_freq_filename, high_freq_resized)
#                     frame_saved = True
#
#                 except KeyError as e:
#                      print(f"Warning [Clip {item_idx}, Frame {i}]: Missing High Freq norm key in config: {e}. Skipping high freq.")
#                 except Exception as e:
#                      print(f"Error processing/saving high frequency map for {img_path}: {e}")
#
#             # Update saved count if anything was successfully saved for this frame
#             if frame_saved:
#                 saved_count += 1
#             else:
#                 skipped_count += 1 # Frame processed but nothing saved due to errors
#
#         except Exception as e:
#             print(f"\n--- Unhandled error processing frame {i} in clip {item_idx} ---")
#             print(f"Image path: {img_path if 'img_path' in locals() else 'N/A'}")
#             print(f"Error: {e}")
#             traceback.print_exc()
#             print("--- End of Frame Error ---")
#             skipped_count += 1
#             continue # Move to the next frame
#
# print("\n--- Saving Loop Finished ---")
# print(f"Total frames processed (attempted): approx {len(predictions) * clip_len_pred}") # Estimate
# print(f"Frames successfully saved (at least one output): {saved_count}")
# print(f"Frames skipped due to errors or warnings: {skipped_count}")
# print(f"Output predictions saved under: {save_path}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# 以下是非常重要预测



# from torch.utils.data import Dataset, DataLoader
# import pytorch_lightning as pl
# from dataloader.New_PolypVideoDataset import SlidingWindowClipSampler
# from argparse import ArgumentParser
# import numpy as np
# from util.util import *  # Assumes build_data, build_model are here
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.callbacks import LearningRateFinder
# import yaml
# import cv2  # Ensure导入cv2
# import os
# import torch
# import warnings
# import traceback  # For detailed error printing
#
# warnings.filterwarnings("ignore")
#
# parser = ArgumentParser()
# parser.add_argument(
#     "--cfg",
#     type=str,
#     default="configs/New_PolypVideoDataset.yaml",  # Ensure this path is correct
#     help="Configuration file to use",
# )
# parser.add_argument(
#     "--output_pred",
#     type=str,
#     default="./pred2",
#     help="Path to save output predictions",
# )
# parser.add_argument(
#     "--checkpoint",
#     type=str,
#     # !!! CRITICAL: ENSURE THIS PATH IS CORRECT AND POINTS TO A WaveletNetPlus CHECKPOINT !!!
#     default="checkpoints/xxx/xx/New_PolypVideoDataset_15/New_PolypVideoDataset_15.ckpt",
#     help="Path to the FULLY TRAINED WaveletNetPlus model checkpoint.",
# )
# parser.add_argument(
#     "--overlay",
#     action='store_true',  # Use action='store_true' for boolean flags
#     help="Generate the overlay image (original image blended with the mask).",
# )
# # Optional: Add a flag to disable saving frequency maps for faster testing
# parser.add_argument(
#     "--no_freq_maps",
#     action='store_true',
#     help="Disable saving of frequency maps.",
# )
#
# args = parser.parse_args()
#
# # --- 1. Load Configuration ---
# print(f"Loading configuration from: {args.cfg}")
# if not os.path.exists(args.cfg):
#     print(f"Error: Configuration file not found at {args.cfg}")
#     exit()
# with open(args.cfg) as f:
#     cfg = yaml.load(f, Loader=yaml.SafeLoader)
# print("Configuration loaded successfully.")
#
# # --- 2. Build Dataset and DataLoader ---
# print("Building validation dataset...")
# try:
#     # Assuming build_data returns (train_set, val_set)
#     _, val_set = build_data(cfg)  # val_set is New_PolypVideoDataset instance
#     if not val_set or len(val_set) == 0:
#         print("Error: Validation dataset is empty or failed to build.")
#         exit()
#     print(f"Validation dataset built. Number of videos: {len(val_set)}")
#     args.class_list = val_set.CLASSES  # Needed for build_model potentially
# except Exception as e:
#     print(f"Error building dataset: {e}")
#     traceback.print_exc()
#     exit()
#
# save_path = os.path.join(args.output_pred, cfg["DATASET"]["dataset"])
# print(f"Predictions will be saved under: {save_path}")
#
# # --- Create DataLoader ---
# clip_len_pred = cfg["DATASET"].get("clip_len_pred", 8)  # Use same clip_len as val_set init or specify
# stride_pred = cfg["DATASET"].get("stride_pred", clip_len_pred)  # Default stride to clip_len if not specified
# print(f"Using clip_len={clip_len_pred}, stride={stride_pred} for prediction sampling.")
#
# sampler_val = SlidingWindowClipSampler(
#     dataset=val_set,
#     clip_len=clip_len_pred,
#     stride=stride_pred,
#     shuffle=False,
#     drop_last=False
# )
#
# # Check sampler length
# try:
#     sampler_len = len(sampler_val)
#     print(f"Sampler created. Expected number of clips: {sampler_len}")
#     if sampler_len == 0:
#         print("Warning: Sampler generated 0 clips. Check dataset/sampler parameters.")
#         # Decide whether to exit or continue (maybe dataset is small?)
#         # exit()
# except Exception as e:
#     print(f"Warning: Could not get sampler length: {e}")
#
# data_loader_val = DataLoader(
#     val_set,
#     sampler=sampler_val,
#     batch_size=1,  # CRITICAL: Must be 1 when using sampler for clips
#     num_workers=cfg["TRAIN"].get("num_workers", 0),  # Get num_workers safely
#     pin_memory=True
# )
# print(f"DataLoader created. Number of batches (clips): {len(data_loader_val)}")
# if len(data_loader_val) == 0:
#     print("Error: DataLoader is empty. Cannot proceed with prediction.")
#     exit()
#
# # --- 3. Build Model ---
# print("Building model...")
# try:
#     # Pass args to build_model if it needs class_list etc.
#     model = build_model(args, cfg)
#     print("Model built successfully.")
# except Exception as e:
#     print(f"Error building model: {e}")
#     traceback.print_exc()
#     exit()
#
# # --- 4. Setup Trainer ---
# print("Setting up PyTorch Lightning Trainer...")
# trainer = pl.Trainer(
#     devices=1,
#     accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     logger=False,  # No logging needed for prediction
#     enable_model_summary=False,
#     enable_progress_bar=True  # Show progress bar during predict
# )
# print(f"Trainer ready. Using accelerator: {trainer.accelerator}")
#
# # --- 5. Perform Prediction ---
# print(f"\n--- Starting prediction using checkpoint: {args.checkpoint} ---")
# if not os.path.exists(args.checkpoint):
#     print(f"Error: Checkpoint file not found at {args.checkpoint}")
#     exit()
#
# predictions = None
# try:
#     predictions = trainer.predict(
#         model, dataloaders=data_loader_val, ckpt_path=args.checkpoint
#     )
# except Exception as e:
#     print(f"\n--- Error during trainer.predict ---")
#     print(f"Exception type: {type(e)}")
#     print(f"Error message: {e}")
#     print("Traceback:")
#     traceback.print_exc()
#     print("--- End of Error ---")
#     predictions = []  # Ensure predictions is a list even on error
#
# # --- CRITICAL CHECK ---
# if predictions is None:
#     print("\nError: trainer.predict returned None. Prediction failed.")
#     exit()
# elif not isinstance(predictions, list):
#     print(f"\nError: trainer.predict returned type {type(predictions)}, expected list.")
#     exit()
# elif len(predictions) == 0:
#     print("\nWarning: trainer.predict returned an empty list []. No predictions were generated.")
#     print("Possible reasons: Empty DataLoader, error inside predict_step, OOM.")
#     exit()
# else:
#     print(f"\nPrediction finished. Got {len(predictions)} prediction items (clips).")
#
# # --- 6. Process and Save Predictions ---
# color_map = {
#     0: (0, 0, 0),       # background Black (RGB)
#     1: (0, 255, 0),     # Polyp Green (RGB)
#     2: (0, 0, 255),     # instrument Blue (RGB)
# }
#
# print("\n--- Starting saving loop... ---")
# saved_count = 0
# skipped_count = 0
# for item_idx, item in enumerate(predictions):
#     if not isinstance(item, dict) or not all(k in item for k in ["preds", "img_paths", "low_freqs", "high_freqs"]):
#         print(f"Warning [Clip {item_idx}]: Unexpected item format. Skipping. Keys: {item.keys() if isinstance(item, dict) else 'Not a dict'}")
#         skipped_count += 1
#         continue
#
#     try:
#         num_frames = item["preds"].shape[0]
#         all_preds = item["preds"].cpu().numpy().astype(np.uint8)  # (T, H, W)
#         all_img_paths = item["img_paths"]                 # List of T paths
#         all_low_freqs = item["low_freqs"].cpu().numpy()   # (T, C, H_in, W_in)
#         all_high_freqs = item["high_freqs"].cpu().numpy()  # (T, C, H_in, W_in)
#     except Exception as e:
#         print(f"Error unpacking data from item {item_idx}: {e}. Skipping clip.")
#         skipped_count += item.get("preds", torch.empty(0)).shape[0]
#         continue
#
#     # --- Data Consistency Check ---
#     if not (len(all_img_paths) == num_frames and all_low_freqs.shape[0] == num_frames and all_high_freqs.shape[0] == num_frames):
#         print(f"Warning [Clip {item_idx}]: Data length mismatch. Frames: {num_frames}, Paths: {len(all_img_paths)}, LowFreq: {all_low_freqs.shape[0]}, HighFreq: {all_high_freqs.shape[0]}. Skipping clip.")
#         skipped_count += num_frames
#         continue
#
#     print(f"Processing clip {item_idx+1}/{len(predictions)}, {num_frames} frames...")
#
#     for i in range(num_frames):
#         frame_saved = False
#         pred = all_preds[i]
#         img_path = all_img_paths[i]
#
#         # 处理 img_path 是列表的情况
#         if isinstance(img_path, list):
#             if len(img_path) == 1 and isinstance(img_path[0], str):
#                 img_path = img_path[0]  # 提取列表中的字符串
#             else:
#                 print(f"Warning [Clip {item_idx}, Frame {i}]: img_path is a list but not a single string: {img_path}. Skipping.")
#                 skipped_count += 1
#                 continue
#         elif not isinstance(img_path, str):
#             print(f"Warning [Clip {item_idx}, Frame {i}]: img_path is not a string: {img_path}. Skipping.")
#             skipped_count += 1
#             continue
#
#         # 规范化路径
#         img_path = os.path.normpath(img_path)
#
#         # 检查文件是否存在
#         if not os.path.exists(img_path):
#             print(f"Warning [Clip {item_idx}, Frame {i}]: File does not exist at {img_path}. Skipping.")
#             skipped_count += 1
#             continue
#
#         orig_img = cv2.imread(img_path)
#         if orig_img is None:
#             print(f"Warning [Clip {item_idx}, Frame {i}]: Failed to load image at {img_path}. Skipping.")
#             skipped_count += 1
#             continue
#
#         # --- Mask Saving Logic ---
#         pred_h, pred_w = pred.shape
#         color_mask_bgr = np.zeros((pred_h, pred_w, 3), dtype=np.uint8)
#         for class_id, color_rgb in color_map.items():
#             color_bgr = tuple(reversed(color_rgb))
#             color_mask_bgr[pred == class_id] = color_bgr
#
#         # 确定保存路径
#         parts = img_path.split(os.sep)
#         subfolder = parts[-2] if len(parts) >= 2 else "unknown_subfolder"
#         file_name = parts[-1]
#         base_name, _ = os.path.splitext(file_name)
#
#         save_dir = os.path.join(save_path, subfolder)
#         os.makedirs(save_dir, exist_ok=True)
#
#
#
#
#         # 保存掩码
#         not_overlay_dir = os.path.join(save_dir, "not_overlay")
#         os.makedirs(not_overlay_dir, exist_ok=True)
#         not_overlay_filename = os.path.join(not_overlay_dir, f"{base_name}_pred.png")
#         cv2.imwrite(not_overlay_filename, color_mask_bgr)
#         frame_saved = True
#
#         # --- Overlay Logic ---
#         if args.overlay:
#             orig_img_resized = cv2.resize(orig_img, (pred_w, pred_h))
#             overlay = cv2.addWeighted(orig_img_resized, 0.6, color_mask_bgr, 0.4, 0)
#             overlay_dir = os.path.join(save_dir, "overlay")
#             os.makedirs(overlay_dir, exist_ok=True)
#             overlay_filename = os.path.join(overlay_dir, f"{base_name}_overlay.png")
#             cv2.imwrite(overlay_filename, overlay)
#             frame_saved = True
#
#         # --- Frequency Map Saving ---
#         if not args.no_freq_maps:
#             mean_low = cfg["DATASET"]["Low_mean"][0] if isinstance(cfg["DATASET"]["Low_mean"], (list, tuple)) else cfg["DATASET"]["Low_mean"]
#             std_low = cfg["DATASET"]["Low_std"][0] if isinstance(cfg["DATASET"]["Low_std"], (list, tuple)) else cfg["DATASET"]["Low_std"]
#             denorm_low_freq = (all_low_freqs[i].squeeze(0) * std_low + mean_low) * 255.0
#             low_freq_resized = cv2.resize(denorm_low_freq, (pred_w, pred_h))
#             low_freq_resized = np.clip(low_freq_resized, 0, 255).astype(np.uint8)
#             low_freq_dir = os.path.join(save_dir, "low_freq")
#             os.makedirs(low_freq_dir, exist_ok=True)
#             low_freq_filename = os.path.join(low_freq_dir, f"{base_name}_low.png")
#             cv2.imwrite(low_freq_filename, low_freq_resized)
#             frame_saved = True
#
#             mean_high = cfg["DATASET"]["High_mean"][0] if isinstance(cfg["DATASET"]["High_mean"], (list, tuple)) else cfg["DATASET"]["High_mean"]
#             std_high = cfg["DATASET"]["High_std"][0] if isinstance(cfg["DATASET"]["High_std"], (list, tuple)) else cfg["DATASET"]["High_std"]
#             denorm_high_freq = (all_high_freqs[i].squeeze(0) * std_high + mean_high) * 255.0
#             high_freq_resized = cv2.resize(denorm_high_freq, (pred_w, pred_h))
#             high_freq_resized = np.clip(high_freq_resized, 0, 255).astype(np.uint8)
#             high_freq_dir = os.path.join(save_dir, "high_freq")
#             os.makedirs(high_freq_dir, exist_ok=True)
#             high_freq_filename = os.path.join(high_freq_dir, f"{base_name}_high.png")
#             cv2.imwrite(high_freq_filename, high_freq_resized)
#             frame_saved = True
#
#         if frame_saved:
#             saved_count += 1
#         else:
#             skipped_count += 1
#
# print("\n--- Saving Loop Finished ---")
# print(f"Total frames processed (attempted): {len(predictions) * clip_len_pred}")
# print(f"Frames successfully saved: {saved_count}")
# print(f"Frames skipped due to errors or warnings: {skipped_count}")
# print(f"Output predictions saved under: {save_path}")


# import sys
# import os
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#     QPushButton, QLabel, QFileDialog, QLineEdit, QCheckBox, QProgressBar,
#     QMessageBox, QTabWidget, QSplitter
# )
# from PyQt5.QtGui import QPixmap, QImage, QColor
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
#
#
# # Assuming these are available from your original script
# # from util.util import build_data, build_model
# # from dataloader.New_PolypVideoDataset import SlidingWindowClipSampler
# # import pytorch_lightning as pl
# # import yaml
# # import torch
# # import warnings
# # import traceback
#
# # --- Placeholder for your original logic ---
# # You'll adapt these to be methods within SegmentationWorker
# def build_data(cfg):
#     # Dummy implementation for illustration
#     print("Building dummy data...")
#
#     class DummyDataset:
#         CLASSES = ["background", "polyp", "instrument"]
#
#         def __len__(self): return 10
#
#         def __getitem__(self, idx): return {}  # Return dummy data
#
#     return None, DummyDataset()
#
#
# def build_model(args, cfg):
#     # Dummy implementation for illustration
#     print("Building dummy model...")
#
#     class DummyModel(pl.LightningModule):
#         def predict_step(self, batch, batch_idx):
#             # Simulate prediction output
#             # (T, H, W) for preds, list of paths, (T, C, H, W) for freqs
#             dummy_preds = np.random.randint(0, 3, (8, 256, 256), dtype=np.uint8)
#             dummy_img_paths = [f"path/to/img_{i}.jpg" for i in range(8)]
#             dummy_low_freqs = np.random.rand(8, 1, 128, 128).astype(np.float32)
#             dummy_high_freqs = np.random.rand(8, 1, 128, 128).astype(np.float32)
#             return {
#                 "preds": torch.from_numpy(dummy_preds),
#                 "img_paths": dummy_img_paths,
#                 "low_freqs": torch.from_numpy(dummy_low_freqs),
#                 "high_freqs": torch.from_numpy(dummy_high_freqs)
#             }
#
#     return DummyModel()
#
#
# # --- End Placeholder ---
#
# # Helper function to convert OpenCV image (numpy array) to QPixmap
# def convert_cv_to_qpixmap(cv_img, target_size=None):
#     if cv_img is None:
#         return QPixmap()
#
#     # Ensure 3 channels for color mask, or handle grayscale
#     if len(cv_img.shape) == 2:  # Grayscale image (e.g., frequency maps)
#         h, w = cv_img.shape
#         bytes_per_line = w
#         q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
#     else:  # Color image (BGR from OpenCV)
#         h, w, ch = cv_img.shape
#         bytes_per_line = ch * w
#         q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
#
#     if target_size:
#         return QPixmap.fromImage(q_img).scaled(target_size[0], target_size[1], Qt.KeepAspectRatio,
#                                                Qt.SmoothTransformation)
#     return QPixmap.fromImage(q_img)
#
#
# class SegmentationWorker(QThread):
#     # Signals to communicate with the GUI thread
#     frame_processed = pyqtSignal(QPixmap, QPixmap, QPixmap, QPixmap, QPixmap, int,
#                                  int)  # orig, mask, overlay, low, high, current, total
#     segmentation_finished = pyqtSignal()
#     progress_update = pyqtSignal(int)
#     status_update = pyqtSignal(str)
#     error_occurred = pyqtSignal(str)
#
#     def __init__(self, config_path, checkpoint_path, output_pred_path,
#                  overlay_enabled, no_freq_maps_enabled, parent=None):
#         super().__init__(parent)
#         self.config_path = config_path
#         self.checkpoint_path = checkpoint_path
#         self.output_pred_path = output_pred_path
#         self.overlay_enabled = overlay_enabled
#         self.no_freq_maps_enabled = no_freq_maps_enabled
#         self._is_running = True
#
#         self.cfg = None
#         self.model = None
#         self.val_set = None
#         self.data_loader_val = None
#         self.color_map = {
#             0: (0, 0, 0),  # background Black (RGB)
#             1: (0, 255, 0),  # Polyp Green (RGB)
#             2: (0, 0, 255),  # instrument Blue (RGB)
#         }
#
#     def stop(self):
#         self._is_running = False
#
#     def run(self):
#         self.status_update.emit("Initializing segmentation worker...")
#         try:
#             # 1. Load Configuration
#             if not os.path.exists(self.config_path):
#                 raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
#             with open(self.config_path) as f:
#                 self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
#
#             # 2. Build Dataset and DataLoader
#             # Create a dummy args object for build_data and build_model if they rely on it
#             # Your original script passes `args` to build_model. Adapt as needed.
#             class DummyArgs:
#                 def __init__(self, cfg_data, checkpoint_path_val):
#                     self.cfg = cfg_data
#                     self.checkpoint = checkpoint_path_val
#                     self.output_pred = self.output_pred_path
#                     self.overlay = self.overlay_enabled
#                     self.no_freq_maps = self.no_freq_maps_enabled
#                     self.class_list = None  # Will be set after val_set is built
#
#             temp_args = DummyArgs(self.cfg, self.checkpoint_path)
#
#             _, self.val_set = build_data(self.cfg)
#             if not self.val_set or len(self.val_set) == 0:
#                 raise ValueError("Validation dataset is empty or failed to build.")
#             temp_args.class_list = self.val_set.CLASSES
#
#             clip_len_pred = self.cfg["DATASET"].get("clip_len_pred", 8)
#             stride_pred = self.cfg["DATASET"].get("stride_pred", clip_len_pred)
#
#             sampler_val = SlidingWindowClipSampler(
#                 dataset=self.val_set,
#                 clip_len=clip_len_pred,
#                 stride=stride_pred,
#                 shuffle=False,
#                 drop_last=False
#             )
#
#             self.data_loader_val = DataLoader(
#                 self.val_set,
#                 sampler=sampler_val,
#                 batch_size=1,  # Critical for clip-based processing
#                 num_workers=self.cfg["TRAIN"].get("num_workers", 0),
#                 pin_memory=True
#             )
#             if len(self.data_loader_val) == 0:
#                 raise ValueError("DataLoader is empty. Cannot proceed with prediction.")
#
#             # 3. Build Model
#             self.model = build_model(temp_args, self.cfg)
#
#             # 4. Setup Trainer
#             trainer = pl.Trainer(
#                 devices=1,
#                 accelerator="gpu" if torch.cuda.is_available() else "cpu",
#                 logger=False,
#                 enable_model_summary=False,
#                 enable_progress_bar=False  # Handled by custom progress signal
#             )
#
#             # 5. Perform Prediction
#             if not os.path.exists(self.checkpoint_path):
#                 raise FileNotFoundError(f"Checkpoint file not found at {self.checkpoint_path}")
#
#             self.status_update.emit(f"Starting prediction with {len(self.data_loader_val)} clips...")
#             predictions = trainer.predict(
#                 self.model, dataloaders=self.data_loader_val, ckpt_path=self.checkpoint_path
#             )
#
#             if not predictions:
#                 self.status_update.emit("Prediction returned no results.")
#                 self.segmentation_finished.emit()
#                 return
#
#             # 6. Process and Save Predictions
#             total_frames_processed = 0
#             for item_idx, item in enumerate(predictions):
#                 if not self._is_running:  # Allow stopping
#                     self.status_update.emit("Segmentation stopped by user.")
#                     break
#
#                 if not isinstance(item, dict) or not all(
#                         k in item for k in ["preds", "img_paths", "low_freqs", "high_freqs"]):
#                     self.error_occurred.emit(f"Warning: Unexpected item format for clip {item_idx}. Skipping.")
#                     continue
#
#                 try:
#                     num_frames = item["preds"].shape[0]
#                     all_preds = item["preds"].cpu().numpy().astype(np.uint8)  # (T, H, W)
#                     all_img_paths = item["img_paths"]  # List of T paths
#                     all_low_freqs = item["low_freqs"].cpu().numpy()  # (T, C, H_in, W_in)
#                     all_high_freqs = item["high_freqs"].cpu().numpy()  # (T, C, H_in, W_in)
#                 except Exception as e:
#                     self.error_occurred.emit(f"Error unpacking data from item {item_idx}: {e}. Skipping clip.")
#                     continue
#
#                 for i in range(num_frames):
#                     if not self._is_running:
#                         self.status_update.emit("Segmentation stopped by user.")
#                         break
#
#                     pred = all_preds[i]
#                     img_path = all_img_paths[i]
#
#                     # Handle img_path being a list
#                     if isinstance(img_path, list):
#                         if len(img_path) == 1 and isinstance(img_path[0], str):
#                             img_path = img_path[0]
#                         else:
#                             self.error_occurred.emit(
#                                 f"Warning: img_path is a list but not a single string: {img_path}. Skipping frame.")
#                             continue
#                     elif not isinstance(img_path, str):
#                         self.error_occurred.emit(f"Warning: img_path is not a string: {img_path}. Skipping frame.")
#                         continue
#
#                     img_path = os.path.normpath(img_path)
#
#                     orig_img = cv2.imread(img_path)
#                     if orig_img is None:
#                         self.error_occurred.emit(
#                             f"Warning: Failed to load original image at {img_path}. Skipping frame.")
#                         continue
#
#                     pred_h, pred_w = pred.shape
#
#                     # Create color mask
#                     color_mask_bgr = np.zeros((pred_h, pred_w, 3), dtype=np.uint8)
#                     for class_id, color_rgb in self.color_map.items():
#                         color_bgr = tuple(reversed(color_rgb))
#                         color_mask_bgr[pred == class_id] = color_bgr
#
#                     # Prepare images for display (QPixmap)
#                     orig_qpix = convert_cv_to_qpixmap(orig_img, target_size=(pred_w, pred_h))
#                     mask_qpix = convert_cv_to_qpixmap(color_mask_bgr)
#
#                     overlay_qpix = QPixmap()
#                     if self.overlay_enabled:
#                         orig_img_resized = cv2.resize(orig_img, (pred_w, pred_h))
#                         overlay_img = cv2.addWeighted(orig_img_resized, 0.6, color_mask_bgr, 0.4, 0)
#                         overlay_qpix = convert_cv_to_qpixmap(overlay_img)
#
#                     low_freq_qpix = QPixmap()
#                     high_freq_qpix = QPixmap()
#                     if not self.no_freq_maps_enabled:
#                         mean_low = self.cfg["DATASET"]["Low_mean"][0] if isinstance(self.cfg["DATASET"]["Low_mean"],
#                                                                                     (list, tuple)) else \
#                         self.cfg["DATASET"]["Low_mean"]
#                         std_low = self.cfg["DATASET"]["Low_std"][0] if isinstance(self.cfg["DATASET"]["Low_std"],
#                                                                                   (list, tuple)) else \
#                         self.cfg["DATASET"]["Low_std"]
#                         denorm_low_freq = (all_low_freqs[i].squeeze(0) * std_low + mean_low) * 255.0
#                         low_freq_resized = cv2.resize(denorm_low_freq, (pred_w, pred_h))
#                         low_freq_resized = np.clip(low_freq_resized, 0, 255).astype(np.uint8)
#                         low_freq_qpix = convert_cv_to_qpixmap(low_freq_resized)
#
#                         mean_high = self.cfg["DATASET"]["High_mean"][0] if isinstance(self.cfg["DATASET"]["High_mean"],
#                                                                                       (list, tuple)) else \
#                         self.cfg["DATASET"]["High_mean"]
#                         std_high = self.cfg["DATASET"]["High_std"][0] if isinstance(self.cfg["DATASET"]["High_std"],
#                                                                                     (list, tuple)) else \
#                         self.cfg["DATASET"]["High_std"]
#                         denorm_high_freq = (all_high_freqs[i].squeeze(0) * std_high + mean_high) * 255.0
#                         high_freq_resized = cv2.resize(denorm_high_freq, (pred_w, pred_h))
#                         high_freq_resized = np.clip(high_freq_resized, 0, 255).astype(np.uint8)
#                         high_freq_qpix = convert_cv_to_qpixmap(high_freq_resized)
#
#                     total_frames = len(predictions) * clip_len_pred  # Approximate total
#                     total_frames_processed += 1
#
#                     self.frame_processed.emit(
#                         orig_qpix, mask_qpix, overlay_qpix, low_freq_qpix, high_freq_qpix,
#                         total_frames_processed, total_frames  # Pass current and total for progress
#                     )
#                     self.progress_update.emit(int(total_frames_processed / total_frames * 100))
#
#                     # You would also save files here as in your original script
#                     # For simplicity, saving is omitted in this GUI example.
#                     # save_dir = os.path.join(self.output_pred_path, ...)
#                     # os.makedirs(save_dir, exist_ok=True)
#                     # cv2.imwrite(os.path.join(save_dir, f"{base_name}_pred.png"), color_mask_bgr)
#                     # ... and so on for overlay, low_freq, high_freq
#
#         except Exception as e:
#             self.error_occurred.emit(f"Segmentation error: {traceback.format_exc()}")
#         finally:
#             self.segmentation_finished.emit()
#             self.status_update.emit("Segmentation process completed or terminated.")
#
#
# class VideoSegmentationApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Polyp Video Segmentation")
#         self.setGeometry(100, 100, 1200, 800)  # Initial window size
#
#         self.worker = None  # Initialize worker
#         self.init_ui()
#
#         self.current_cfg_path = "configs/New_PolypVideoDataset.yaml"
#         self.current_checkpoint_path = "checkpoints/xxx/xx/New_PolypVideoDataset_15/New_PolypVideoDataset_15.ckpt"
#         self.current_output_dir = "./pred1"
#
#         self.update_path_labels()
#
#         # Apply some styling
#         self.setStyleSheet("""
#             QMainWindow {
#                 background-color: #2e2e2e;
#                 color: #e0e0e0;
#             }
#             QPushButton {
#                 background-color: #4CAF50;
#                 color: white;
#                 border-radius: 5px;
#                 padding: 8px 15px;
#                 font-weight: bold;
#             }
#             QPushButton:hover {
#                 background-color: #45a049;
#             }
#             QLineEdit {
#                 background-color: #3e3e3e;
#                 border: 1px solid #555;
#                 border-radius: 3px;
#                 padding: 5px;
#                 color: #e0e0e0;
#             }
#             QLabel {
#                 color: #e0e0e0;
#             }
#             QCheckBox {
#                 color: #e0e0e0;
#             }
#             QProgressBar {
#                 text-align: center;
#                 color: black;
#                 background-color: #444;
#                 border-radius: 5px;
#             }
#             QProgressBar::chunk {
#                 background-color: #007bff;
#                 border-radius: 5px;
#             }
#             QTabWidget::pane {
#                 border: 1px solid #555;
#             }
#             QTabBar::tab {
#                 background: #3e3e3e;
#                 color: #e0e0e0;
#                 padding: 8px;
#                 border-top-left-radius: 5px;
#                 border-top-right-radius: 5px;
#             }
#             QTabBar::tab:selected {
#                 background: #555;
#                 color: white;
#             }
#         """)
#
#     def init_ui(self):
#         # Create central widget and main layout
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
#         main_layout = QHBoxLayout(central_widget)
#
#         # --- Control Panel (Left Side) ---
#         control_panel = QWidget()
#         control_layout = QVBoxLayout(control_panel)
#         control_panel.setFixedWidth(300)  # Fixed width for control panel
#
#         # File Selection
#         control_layout.addWidget(QLabel("<h4>Configuration & Files</h4>"))
#         self.cfg_path_label = QLineEdit()
#         self.cfg_path_label.setReadOnly(True)
#         btn_cfg = QPushButton("Select Config File")
#         btn_cfg.clicked.connect(self.select_config_file)
#         control_layout.addWidget(self.cfg_path_label)
#         control_layout.addWidget(btn_cfg)
#
#         self.checkpoint_path_label = QLineEdit()
#         self.checkpoint_path_label.setReadOnly(True)
#         btn_checkpoint = QPushButton("Select Checkpoint")
#         btn_checkpoint.clicked.connect(self.select_checkpoint_file)
#         control_layout.addWidget(self.checkpoint_path_label)
#         control_layout.addWidget(btn_checkpoint)
#
#         self.output_dir_label = QLineEdit()
#         self.output_dir_label.setReadOnly(True)
#         btn_output_dir = QPushButton("Select Output Directory")
#         btn_output_dir.clicked.connect(self.select_output_directory)
#         control_layout.addWidget(self.output_dir_label)
#         control_layout.addWidget(btn_output_dir)
#
#         # Options
#         control_layout.addWidget(QLabel("<h4>Segmentation Options</h4>"))
#         self.checkbox_overlay = QCheckBox("Generate Overlay Image")
#         self.checkbox_overlay.setChecked(True)  # Default from args.overlay
#         control_layout.addWidget(self.checkbox_overlay)
#
#         self.checkbox_no_freq_maps = QCheckBox("Disable Frequency Map Saving")
#         self.checkbox_no_freq_maps.setChecked(False)  # Default from not args.no_freq_maps
#         control_layout.addWidget(self.checkbox_no_freq_maps)
#
#         # Action Buttons
#         control_layout.addStretch()  # Push buttons to bottom
#         self.btn_start = QPushButton("Start Segmentation")
#         self.btn_start.clicked.connect(self.start_segmentation)
#         control_layout.addWidget(self.btn_start)
#
#         self.btn_stop = QPushButton("Stop Segmentation")
#         self.btn_stop.clicked.connect(self.stop_segmentation)
#         self.btn_stop.setEnabled(False)  # Initially disabled
#         control_layout.addWidget(self.btn_stop)
#
#         main_layout.addWidget(control_panel)
#
#         # --- Display Area (Right Side) ---
#         display_area = QWidget()
#         display_layout = QVBoxLayout(display_area)
#
#         # Tab Widget for different views
#         self.tab_widget = QTabWidget()
#         display_layout.addWidget(self.tab_widget)
#
#         # Original Image Tab
#         self.orig_label = QLabel("Original Image")
#         self.orig_label.setAlignment(Qt.AlignCenter)
#         self.orig_label.setScaledContents(True)
#         self.tab_widget.addTab(self.orig_label, "Original")
#
#         # Predicted Mask Tab
#         self.mask_label = QLabel("Predicted Mask")
#         self.mask_label.setAlignment(Qt.AlignCenter)
#         self.mask_label.setScaledContents(True)
#         self.tab_widget.addTab(self.mask_label, "Mask")
#
#         # Overlay Tab
#         self.overlay_label = QLabel("Overlay")
#         self.overlay_label.setAlignment(Qt.AlignCenter)
#         self.overlay_label.setScaledContents(True)
#         self.tab_widget.addTab(self.overlay_label, "Overlay")
#
#         # Low Frequency Tab
#         self.low_freq_label = QLabel("Low Frequency Map")
#         self.low_freq_label.setAlignment(Qt.AlignCenter)
#         self.low_freq_label.setScaledContents(True)
#         self.tab_widget.addTab(self.low_freq_label, "Low Freq")
#
#         # High Frequency Tab
#         self.high_freq_label = QLabel("High Frequency Map")
#         self.high_freq_label.setAlignment(Qt.AlignCenter)
#         self.high_freq_label.setScaledContents(True)
#         self.tab_widget.addTab(self.high_freq_label, "High Freq")
#
#         # Progress Bar and Status Bar
#         self.progress_bar = QProgressBar()
#         self.progress_bar.setValue(0)
#         display_layout.addWidget(self.progress_bar)
#
#         self.status_label = QLabel("Ready.")
#         self.statusBar().addWidget(self.status_label)
#
#         main_layout.addWidget(display_area)
#
#     def update_path_labels(self):
#         self.cfg_path_label.setText(self.current_cfg_path)
#         self.checkpoint_path_label.setText(self.current_checkpoint_path)
#         self.output_dir_label.setText(self.current_output_dir)
#
#     @pyqtSlot()
#     def select_config_file(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "Select Configuration File", "",
#                                                    "YAML Files (*.yaml);;All Files (*)")
#         if file_path:
#             self.current_cfg_path = file_path
#             self.cfg_path_label.setText(file_path)
#
#     @pyqtSlot()
#     def select_checkpoint_file(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "",
#                                                    "Checkpoint Files (*.ckpt);;All Files (*)")
#         if file_path:
#             self.current_checkpoint_path = file_path
#             self.checkpoint_path_label.setText(file_path)
#
#     @pyqtSlot()
#     def select_output_directory(self):
#         dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
#         if dir_path:
#             self.current_output_dir = dir_path
#             self.output_dir_label.setText(dir_path)
#
#     @pyqtSlot()
#     def start_segmentation(self):
#         if self.worker and self.worker.isRunning():
#             QMessageBox.warning(self, "Warning", "Segmentation is already running.")
#             return
#
#         # Clear previous results
#         self.orig_label.clear()
#         self.mask_label.clear()
#         self.overlay_label.clear()
#         self.low_freq_label.clear()
#         self.high_freq_label.clear()
#         self.progress_bar.setValue(0)
#         self.status_label.setText("Starting segmentation...")
#
#         self.btn_start.setEnabled(False)
#         self.btn_stop.setEnabled(True)
#
#         self.worker = SegmentationWorker(
#             self.current_cfg_path,
#             self.current_checkpoint_path,
#             self.current_output_dir,
#             self.checkbox_overlay.isChecked(),
#             self.checkbox_no_freq_maps.isChecked()
#         )
#         self.worker.frame_processed.connect(self.update_display)
#         self.worker.segmentation_finished.connect(self.segmentation_finished)
#         self.worker.progress_update.connect(self.progress_bar.setValue)
#         self.worker.status_update.connect(self.status_label.setText)
#         self.worker.error_occurred.connect(self.show_error_message)
#
#         self.worker.start()  # Start the worker thread
#
#     @pyqtSlot()
#     def stop_segmentation(self):
#         if self.worker and self.worker.isRunning():
#             self.worker.stop()
#             self.btn_stop.setEnabled(False)  # Will be re-enabled by finished signal
#             self.status_label.setText("Stopping segmentation, please wait...")
#         else:
#             QMessageBox.information(self, "Info", "No segmentation process is running.")
#
#     @pyqtSlot(QPixmap, QPixmap, QPixmap, QPixmap, QPixmap, int, int)
#     def update_display(self, orig_qpix, mask_qpix, overlay_qpix, low_freq_qpix, high_freq_qpix, current_frame,
#                        total_frames):
#         self.orig_label.setPixmap(orig_qpix)
#         self.mask_label.setPixmap(mask_qpix)
#         if self.checkbox_overlay.isChecked():
#             self.overlay_label.setPixmap(overlay_qpix)
#         else:
#             self.overlay_label.clear()  # Clear if overlay is not requested
#
#         if not self.checkbox_no_freq_maps.isChecked():
#             self.low_freq_label.setPixmap(low_freq_qpix)
#             self.high_freq_label.setPixmap(high_freq_qpix)
#         else:
#             self.low_freq_label.clear()
#             self.high_freq_label.clear()
#
#         self.status_label.setText(f"Processing frame {current_frame}/{total_frames}")
#
#     @pyqtSlot()
#     def segmentation_finished(self):
#         self.status_label.setText("Segmentation process complete.")
#         self.btn_start.setEnabled(True)
#         self.btn_stop.setEnabled(False)
#         self.progress_bar.setValue(100)
#         if self.worker:
#             self.worker.quit()  # Safely quit the thread
#             self.worker.wait()  # Wait for it to finish
#
#     @pyqtSlot(str)
#     def show_error_message(self, message):
#         QMessageBox.critical(self, "Error", message)
#         self.status_label.setText("Error occurred.")
#         self.btn_start.setEnabled(True)
#         self.btn_stop.setEnabled(False)
#         self.progress_bar.setValue(0)
#         if self.worker:
#             self.worker.quit()
#             self.worker.wait()
#
#     def closeEvent(self, event):
#         if self.worker and self.worker.isRunning():
#             reply = QMessageBox.question(self, 'Confirm Exit',
#                                          "Segmentation is still running. Are you sure you want to exit?",
#                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
#             if reply == QMessageBox.Yes:
#                 self.worker.stop()
#                 self.worker.wait(5000)  # Wait up to 5 seconds for thread to finish
#                 if self.worker.isRunning():
#                     print("Worker thread did not terminate, forcing exit.")
#                 event.accept()
#             else:
#                 event.ignore()
#         else:
#             event.accept()
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = VideoSegmentationApp()
#     window.show()
#     sys.exit(app.exec_())











































# import sys
# import os
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#     QPushButton, QLabel, QFileDialog, QLineEdit, QCheckBox, QProgressBar,
#     QMessageBox, QTabWidget, QSplitter
# )
# from PyQt5.QtGui import QPixmap, QImage, QColor
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
#
# # --- Placeholder for your original logic ---
# # These are dummy implementations for the GUI example.
# # You MUST replace these with your actual functions from your project.
#
# # Mock PyTorch Lightning and other dependencies for the dummy setup
# try:
#     import pytorch_lightning as pl
#     import torch
#     import yaml
#     import warnings
#     from torch.utils.data import DataLoader
# except ImportError:
#     # Fallback for systems without PyTorch/Lightning installed for basic GUI testing
#     print("PyTorch Lightning or Torch not found. Running with basic dummy placeholders.")
#
#
#     class DummyLightningModule:
#         def predict_step(self, batch, batch_idx):
#             pass  # No actual prediction logic
#
#
#     class DummyTrainer:
#         def __init__(self, devices, accelerator, logger, enable_model_summary, enable_progress_bar):
#             self.accelerator = accelerator
#
#         def predict(self, model, dataloaders, ckpt_path):
#             # Simulate a list of prediction outputs
#             num_clips = 5  # Simulate 5 clips
#             frames_per_clip = 8
#             dummy_preds_list = []
#             for _ in range(num_clips):
#                 dummy_preds = np.random.randint(0, 3, (frames_per_clip, 256, 256), dtype=np.uint8)
#                 dummy_img_paths = [
#                     os.path.join("dummy_data", "video_001", f"frame_{i:04d}.jpg")
#                     for i in range(frames_per_clip)
#                 ]
#                 # Ensure dummy image files exist
#                 for path in dummy_img_paths:
#                     os.makedirs(os.path.dirname(path), exist_ok=True)
#                     if not os.path.exists(path):
#                         dummy_img_content = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
#                         cv2.imwrite(path, dummy_img_content)
#                 dummy_low_freqs = np.random.rand(frames_per_clip, 1, 128, 128).astype(np.float32)
#                 dummy_high_freqs = np.random.rand(frames_per_clip, 1, 128, 128).astype(np.float32)
#                 dummy_preds_list.append({
#                     "preds": torch.from_numpy(dummy_preds) if 'torch' in sys.modules else dummy_preds,
#                     "img_paths": dummy_img_paths,
#                     "low_freqs": torch.from_numpy(dummy_low_freqs) if 'torch' in sys.modules else dummy_low_freqs,
#                     "high_freqs": torch.from_numpy(dummy_high_freqs) if 'torch' in sys.modules else dummy_high_freqs
#                 })
#             return dummy_preds_list
#
#
#     pl = type('pl_module', (), {'LightningModule': DummyLightningModule, 'Trainer': DummyTrainer})()
#     torch = type('torch_module', (),
#                  {'cuda': type('cuda_module', (), {'is_available': lambda: False})(), 'from_numpy': lambda x: x})()
#     yaml = type('yaml_module', (), {'load': lambda f, Loader: {
#         'DATASET': {'Low_mean': [0.5], 'Low_std': [0.5], 'High_mean': [0.5], 'High_std': [0.5], 'clip_len_pred': 8,
#                     'stride_pred': 8}, 'TRAIN': {'num_workers': 0}}})()
#     warnings = type('warnings_module', (), {'filterwarnings': lambda x: None})()
#     DataLoader = type('DataLoader_class', (),
#                       {'__init__': lambda self, dataset, sampler, batch_size, num_workers, pin_memory: None,
#                        '__len__': lambda self: 5})()
#
# import traceback
#
# warnings.filterwarnings("ignore")  # To suppress warnings from your original script
#
#
# def build_data(cfg):
#     print("Building dummy data...")
#
#     class DummyDataset:
#         CLASSES = ["background", "polyp", "instrument"]
#
#         def __len__(self): return 100  # Increased for more meaningful progress bar
#
#         def __getitem__(self, idx):
#             # Return dummy data. In a real scenario, this would load frames.
#             # We're simulating img_paths in the predict_step directly for this dummy.
#             return {}
#
#     return None, DummyDataset()
#
#
# def build_model(args, cfg):
#     print("Building dummy model...")
#
#     class DummyModel(pl.LightningModule):
#         # Override predict_step to return dummy data for GUI testing
#         def predict_step(self, batch, batch_idx):
#             # Simulate prediction output for a single clip
#             frames_in_clip = args.cfg["DATASET"].get("clip_len_pred", 8)
#
#             dummy_preds = np.random.randint(0, 3, (frames_in_clip, 256, 256), dtype=np.uint8)
#
#             # Use specific dummy paths to ensure they exist for cv2.imread
#             dummy_img_paths = [
#                 os.path.join("dummy_data", "video_001", f"frame_{batch_idx * frames_in_clip + i:04d}.jpg")
#                 for i in range(frames_in_clip)
#             ]
#             # Create dummy image files if they don't exist
#             for path in dummy_img_paths:
#                 os.makedirs(os.path.dirname(path), exist_ok=True)
#                 if not os.path.exists(path):
#                     # Create a simple colored square image for visual clarity
#                     dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
#                     color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
#                     cv2.rectangle(dummy_img, (50, 50), (200, 200), color, -1)
#                     cv2.putText(dummy_img, f"Frame {os.path.basename(path).split('.')[0]}", (60, 120),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                     cv2.imwrite(path, dummy_img)
#
#             dummy_low_freqs = np.random.rand(frames_in_clip, 1, 128, 128).astype(np.float32)
#             dummy_high_freqs = np.random.rand(frames_in_clip, 1, 128, 128).astype(np.float32)
#
#             return {
#                 "preds": torch.from_numpy(dummy_preds) if 'torch' in sys.modules else dummy_preds,
#                 "img_paths": dummy_img_paths,
#                 "low_freqs": torch.from_numpy(dummy_low_freqs) if 'torch' in sys.modules else dummy_low_freqs,
#                 "high_freqs": torch.from_numpy(dummy_high_freqs) if 'torch' in sys.modules else dummy_high_freqs
#             }
#
#     return DummyModel()
#
#
# # Mock SlidingWindowClipSampler if you don't have it immediately
# class SlidingWindowClipSampler:
#     def __init__(self, dataset, clip_len, stride, shuffle, drop_last):
#         self.dataset = dataset
#         self.clip_len = clip_len
#         self.stride = stride
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#         # Simulate clips based on dataset length
#         self._length = (len(dataset) - clip_len) // stride + 1 if len(dataset) >= clip_len else 0
#         if self._length < 0: self._length = 0  # Ensure non-negative
#
#     def __iter__(self):
#         # This would yield indices for actual clips
#         for i in range(0, len(self.dataset) - self.clip_len + 1, self.stride):
#             yield list(range(i, i + self.clip_len))
#
#     def __len__(self):
#         return self._length
#
#
# # --- End Placeholder ---
#
# # Helper function to convert OpenCV image (numpy array) to QPixmap
# def convert_cv_to_qpixmap(cv_img, target_size=None):
#     if cv_img is None or cv_img.size == 0:
#         return QPixmap()
#
#     # Ensure 3 channels for color mask, or handle grayscale
#     if len(cv_img.shape) == 2:  # Grayscale image (e.g., frequency maps)
#         h, w = cv_img.shape
#         bytes_per_line = w
#         # QImage.Format_Grayscale8 only works for single channel, 8-bit images
#         q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
#     else:  # Color image (BGR from OpenCV)
#         h, w, ch = cv_img.shape
#         if ch == 1:  # Convert grayscale CV_8UC1 to QImage.Format_Grayscale8
#             bytes_per_line = w
#             q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
#         else:  # Assume BGR (3 channels) for QImage.Format_BGR888
#             bytes_per_line = ch * w
#             q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
#
#     if target_size:
#         return QPixmap.fromImage(q_img).scaled(target_size[0], target_size[1], Qt.KeepAspectRatio,
#                                                Qt.SmoothTransformation)
#     return QPixmap.fromImage(q_img)
#
#
# class SegmentationWorker(QThread):
#     # Signals to communicate with the GUI thread
#     # 信号用于与GUI线程通信
#     frame_processed = pyqtSignal(QPixmap, QPixmap, QPixmap, QPixmap, QPixmap, int,
#                                  int)  # orig, mask, overlay, low, high, current, total
#     segmentation_finished = pyqtSignal()
#     progress_update = pyqtSignal(int)
#     status_update = pyqtSignal(str)
#     error_occurred = pyqtSignal(str)
#
#     def __init__(self, config_path, checkpoint_path, output_pred_path,
#                  overlay_enabled, no_freq_maps_enabled, parent=None):
#         super().__init__(parent)
#         self.config_path = config_path
#         self.checkpoint_path = checkpoint_path
#         self.output_pred_path = output_pred_path
#         self.overlay_enabled = overlay_enabled
#         self.no_freq_maps_enabled = no_freq_maps_enabled
#         self._is_running = True
#
#         self.cfg = None
#         self.model = None
#         self.val_set = None
#         self.data_loader_val = None
#         self.color_map = {
#             0: (0, 0, 0),  # background Black (RGB)
#             1: (0, 255, 0),  # Polyp Green (RGB)
#             2: (0, 0, 255),  # instrument Blue (RGB)
#         }
#
#     def stop(self):
#         # 停止线程
#         self._is_running = False
#
#     def run(self):
#         # 线程运行的主逻辑
#         self.status_update.emit("正在初始化分割工作器...")
#         try:
#             # 1. Load Configuration (加载配置)
#             if not os.path.exists(self.config_path):
#                 # For dummy run, create a dummy config if it doesn't exist
#                 # This ensures the dummy setup can always run.
#                 os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
#                 dummy_cfg_content = """
# DATASET:
#   Low_mean: [0.5]
#   Low_std: [0.5]
#   High_mean: [0.5]
#   High_std: [0.5]
#   clip_len_pred: 8
#   stride_pred: 8
# TRAIN:
#   num_workers: 0
#                 """
#                 with open(self.config_path, "w") as f:
#                     f.write(dummy_cfg_content)
#                 self.status_update.emit(f"警告: 配置文件未找到，已创建虚拟配置文件: {self.config_path}")
#
#             with open(self.config_path) as f:
#                 self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
#             self.status_update.emit("配置加载成功。")
#
#             # 2. Build Dataset and DataLoader (构建数据集和数据加载器)
#             class DummyArgs:  # 为build_data和build_model创建虚拟参数对象
#                 def __init__(self, cfg_data, checkpoint_path_val, output_pred_path_val, overlay_enabled_val,
#                              no_freq_maps_enabled_val):
#                     self.cfg = cfg_data
#                     self.checkpoint = checkpoint_path_val
#                     self.output_pred = output_pred_path_val
#                     self.overlay = overlay_enabled_val
#                     self.no_freq_maps = no_freq_maps_enabled_val
#                     self.class_list = None  # Will be set after val_set is built
#
#             temp_args = DummyArgs(self.cfg, self.checkpoint_path, self.output_pred_path, self.overlay_enabled,
#                                   self.no_freq_maps_enabled)
#
#             self.status_update.emit("正在构建验证数据集...")
#             _, self.val_set = build_data(self.cfg)
#             if not self.val_set or len(self.val_set) == 0:
#                 raise ValueError("验证数据集为空或构建失败。")
#             temp_args.class_list = self.val_set.CLASSES
#             self.status_update.emit(f"验证数据集构建完成。模拟视频数量: {len(self.val_set)}")
#
#             clip_len_pred = self.cfg["DATASET"].get("clip_len_pred", 8)
#             stride_pred = self.cfg["DATASET"].get("stride_pred", clip_len_pred)
#             self.status_update.emit(f"预测使用 clip_len={clip_len_pred}, stride={stride_pred}。")
#
#             sampler_val = SlidingWindowClipSampler(
#                 dataset=self.val_set,
#                 clip_len=clip_len_pred,
#                 stride=stride_pred,
#                 shuffle=False,
#                 drop_last=False
#             )
#
#             try:
#                 sampler_len = len(sampler_val)
#                 self.status_update.emit(f"采样器已创建。预期剪辑数量: {sampler_len}")
#                 if sampler_len == 0:
#                     self.error_occurred.emit("警告: 采样器生成0个剪辑。请检查数据集/采样器参数。")
#                     self.segmentation_finished.emit()  # Finish if no clips
#                     return
#             except Exception as e:
#                 self.error_occurred.emit(f"警告: 无法获取采样器长度: {e}")
#                 self.segmentation_finished.emit()  # Finish on sampler error
#                 return
#
#             self.data_loader_val = DataLoader(
#                 self.val_set,
#                 sampler=sampler_val,
#                 batch_size=1,  # Critical for clip-based processing
#                 num_workers=self.cfg["TRAIN"].get("num_workers", 0),
#                 pin_memory=True
#             )
#             if len(self.data_loader_val) == 0:
#                 raise ValueError("数据加载器为空。无法进行预测。")
#             self.status_update.emit(f"数据加载器已创建。批次（剪辑）数量: {len(self.data_loader_val)}")
#
#             # 3. Build Model (构建模型)
#             self.status_update.emit("正在构建模型...")
#             self.model = build_model(temp_args, self.cfg)
#             self.status_update.emit("模型构建成功。")
#
#             # 4. Setup Trainer (设置PyTorch Lightning Trainer)
#             self.status_update.emit("正在设置PyTorch Lightning Trainer...")
#             trainer = pl.Trainer(
#                 devices=1,
#                 accelerator="gpu" if torch.cuda.is_available() else "cpu",
#                 logger=False,
#                 enable_model_summary=False,
#                 enable_progress_bar=False  # Handled by custom progress signal
#             )
#             self.status_update.emit(f"Trainer已就绪。使用加速器: {trainer.accelerator}")
#
#             # 5. Perform Prediction (执行预测)
#             self.status_update.emit(f"\n--- 正在使用检查点进行预测: {self.checkpoint_path} ---")
#
#             # For dummy run, create a dummy checkpoint file if it doesn't exist
#             if not os.path.exists(self.checkpoint_path):
#                 os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
#                 with open(self.checkpoint_path, "w") as f:
#                     f.write("DUMMY_CHECKPOINT_CONTENT")  # Just create an empty file
#                 self.status_update.emit(f"警告: 检查点文件未找到，已创建虚拟检查点: {self.checkpoint_path}")
#
#             predictions = trainer.predict(
#                 self.model, dataloaders=self.data_loader_val, ckpt_path=self.checkpoint_path
#             )
#
#             if predictions is None:
#                 raise RuntimeError("trainer.predict 返回 None。预测失败。")
#             elif not isinstance(predictions, list):
#                 raise TypeError(f"trainer.predict 返回类型 {type(predictions)}, 预期为列表。")
#             elif len(predictions) == 0:
#                 self.error_occurred.emit("警告: trainer.predict 返回空列表。没有生成预测结果。")
#                 self.segmentation_finished.emit()
#                 return
#             else:
#                 self.status_update.emit(f"\n预测完成。获得 {len(predictions)} 个预测项（剪辑）。")
#
#             # 6. Process and Save Predictions (处理并保存预测结果)
#             self.status_update.emit("\n--- 正在开始保存循环... ---")
#
#             # Recalculate total_frames_to_process based on actual predictions for accuracy
#             total_frames_to_process = sum([item["preds"].shape[0] for item in predictions if "preds" in item])
#             if total_frames_to_process == 0:
#                 self.status_update.emit("没有要处理的帧。")
#                 self.segmentation_finished.emit()
#                 return
#
#             processed_frames_count = 0
#
#             for item_idx, item in enumerate(predictions):
#                 if not self._is_running:  # 允许停止
#                     self.status_update.emit("用户已停止分割。")
#                     break
#
#                 if not isinstance(item, dict) or not all(
#                         k in item for k in ["preds", "img_paths", "low_freqs", "high_freqs"]):
#                     self.error_occurred.emit(f"警告 [剪辑 {item_idx}]: 意外的项目格式。跳过。")
#                     continue
#
#                 try:
#                     num_frames = item["preds"].shape[0]
#                     # Ensure conversion from torch tensor if necessary
#                     all_preds = item["preds"].cpu().numpy().astype(np.uint8) if 'torch' in sys.modules and isinstance(
#                         item["preds"], torch.Tensor) else item["preds"].astype(np.uint8)
#                     all_img_paths = item["img_paths"]  # List of T paths
#                     all_low_freqs = item["low_freqs"].cpu().numpy() if 'torch' in sys.modules and isinstance(
#                         item["low_freqs"], torch.Tensor) else item["low_freqs"]  # (T, C, H_in, W_in)
#                     all_high_freqs = item["high_freqs"].cpu().numpy() if 'torch' in sys.modules and isinstance(
#                         item["high_freqs"], torch.Tensor) else item["high_freqs"]  # (T, C, H_in, W_in)
#                 except Exception as e:
#                     self.error_occurred.emit(f"错误: 从项目 {item_idx} 解包数据失败: {e}。跳过剪辑。")
#                     continue
#
#                 if not (len(all_img_paths) == num_frames and all_low_freqs.shape[0] == num_frames and
#                         all_high_freqs.shape[0] == num_frames):
#                     self.error_occurred.emit(f"警告 [剪辑 {item_idx}]: 数据长度不匹配。跳过剪辑。")
#                     continue
#
#                 self.status_update.emit(f"正在处理剪辑 {item_idx + 1}/{len(predictions)}, 共 {num_frames} 帧...")
#
#                 for i in range(num_frames):
#                     if not self._is_running:
#                         self.status_update.emit("用户已停止分割。")
#                         break
#
#                     pred = all_preds[i]
#                     img_path = all_img_paths[i]
#
#                     # 处理 img_path 是列表的情况
#                     if isinstance(img_path, list):
#                         if len(img_path) == 1 and isinstance(img_path[0], str):
#                             img_path = img_path[0]
#                         else:
#                             self.error_occurred.emit(
#                                 f"警告 [剪辑 {item_idx}, 帧 {i}]: img_path 是列表但不是单个字符串: {img_path}。跳过。")
#                             processed_frames_count += 1
#                             continue
#                     elif not isinstance(img_path, str):
#                         self.error_occurred.emit(
#                             f"警告 [剪辑 {item_idx}, 帧 {i}]: img_path 不是字符串: {img_path}。跳过。")
#                         processed_frames_count += 1
#                         continue
#
#                     # 规范化路径
#                     img_path = os.path.normpath(img_path)
#
#                     orig_img = cv2.imread(img_path)
#                     if orig_img is None:
#                         self.error_occurred.emit(
#                             f"警告 [剪辑 {item_idx}, 帧 {i}]: 无法加载图像: {img_path}。请检查文件是否存在和损坏。跳过。")
#                         processed_frames_count += 1
#                         continue
#
#                     pred_h, pred_w = pred.shape
#
#                     # Create color mask (创建彩色掩码)
#                     color_mask_bgr = np.zeros((pred_h, pred_w, 3), dtype=np.uint8)
#                     for class_id, color_rgb in self.color_map.items():
#                         color_bgr = tuple(reversed(color_rgb))  # RGB to BGR
#                         color_mask_bgr[pred == class_id] = color_bgr
#
#                     # Prepare images for display (QPixmap) (准备图像用于显示)
#                     # Rescale images to fit label size for better visual consistency
#                     # The actual size will be adjusted by setScaledContents(True) on QLabel
#                     display_w, display_h = 300, 300  # A reasonable default size for display
#
#                     orig_qpix = convert_cv_to_qpixmap(orig_img, target_size=(display_w, display_h))
#                     mask_qpix = convert_cv_to_qpixmap(color_mask_bgr, target_size=(display_w, display_h))
#
#                     overlay_qpix = QPixmap()
#                     if self.overlay_enabled:
#                         # Resize original image to prediction size for overlay
#                         orig_img_resized = cv2.resize(orig_img, (pred_w, pred_h))
#                         overlay_img = cv2.addWeighted(orig_img_resized, 0.6, color_mask_bgr, 0.4, 0)
#                         overlay_qpix = convert_cv_to_qpixmap(overlay_img, target_size=(display_w, display_h))
#
#                     low_freq_qpix = QPixmap()
#                     high_freq_qpix = QPixmap()
#                     if not self.no_freq_maps_enabled:
#                         # Denormalize and resize frequency maps
#                         mean_low = self.cfg["DATASET"]["Low_mean"][0] if isinstance(self.cfg["DATASET"]["Low_mean"],
#                                                                                     (list, tuple)) else \
#                         self.cfg["DATASET"]["Low_mean"]
#                         std_low = self.cfg["DATASET"]["Low_std"][0] if isinstance(self.cfg["DATASET"]["Low_std"],
#                                                                                   (list, tuple)) else \
#                         self.cfg["DATASET"]["Low_std"]
#
#                         # Ensure all_low_freqs[i] has the correct shape for squeeze
#                         low_freq_data = all_low_freqs[i].squeeze(0) if all_low_freqs[i].ndim > 2 else all_low_freqs[i]
#                         denorm_low_freq = (low_freq_data * std_low + mean_low) * 255.0
#                         low_freq_resized = cv2.resize(denorm_low_freq, (pred_w, pred_h))
#                         low_freq_resized = np.clip(low_freq_resized, 0, 255).astype(np.uint8)
#                         low_freq_qpix = convert_cv_to_qpixmap(low_freq_resized, target_size=(display_w, display_h))
#
#                         mean_high = self.cfg["DATASET"]["High_mean"][0] if isinstance(self.cfg["DATASET"]["High_mean"],
#                                                                                       (list, tuple)) else \
#                         self.cfg["DATASET"]["High_mean"]
#                         std_high = self.cfg["DATASET"]["High_std"][0] if isinstance(self.cfg["DATASET"]["High_std"],
#                                                                                     (list, tuple)) else \
#                         self.cfg["DATASET"]["High_std"]
#
#                         high_freq_data = all_high_freqs[i].squeeze(0) if all_high_freqs[i].ndim > 2 else all_high_freqs[
#                             i]
#                         denorm_high_freq = (high_freq_data * std_high + mean_high) * 255.0
#                         high_freq_resized = cv2.resize(denorm_high_freq, (pred_w, pred_h))
#                         high_freq_resized = np.clip(high_freq_resized, 0, 255).astype(np.uint8)
#                         high_freq_qpix = convert_cv_to_qpixmap(high_freq_resized, target_size=(display_w, display_h))
#
#                     processed_frames_count += 1
#                     # Ensure percentage doesn't exceed 100
#                     percentage = int((processed_frames_count / total_frames_to_process) * 100)
#                     if percentage > 100: percentage = 100
#
#                     self.frame_processed.emit(
#                         orig_qpix, mask_qpix, overlay_qpix, low_freq_qpix, high_freq_qpix,
#                         processed_frames_count, total_frames_to_process  # Pass current and total for progress
#                     )
#                     self.progress_update.emit(percentage)
#
#                     # --- File Saving Logic (文件保存逻辑) ---
#                     # You would re-implement your file saving logic here from the original script
#                     # For simplicity, saving is omitted in this GUI example.
#                     # Example of how you might structure saving if needed:
#                     # parts = img_path.split(os.sep)
#                     # video_subfolder = parts[-2] if len(parts) >= 2 else "unknown_video"
#                     # frame_name = os.path.splitext(parts[-1])[0]
#                     #
#                     # output_sub_dir = os.path.join(self.output_pred_path, video_subfolder)
#                     # os.makedirs(output_sub_dir, exist_ok=True)
#                     #
#                     # if self.overlay_enabled:
#                     #     cv2.imwrite(os.path.join(output_sub_dir, f"{frame_name}_overlay.png"), overlay_img)
#                     # if not self.no_freq_maps_enabled:
#                     #     cv2.imwrite(os.path.join(output_sub_dir, f"{frame_name}_low_freq.png"), low_freq_resized)
#                     #     cv2.imwrite(os.path.join(output_sub_dir, f"{frame_name}_high_freq.png"), high_freq_resized)
#                     # cv2.imwrite(os.path.join(output_sub_dir, f"{frame_name}_mask.png"), color_mask_bgr)
#
#         except Exception as e:
#             # 捕获所有异常并发出错误信号
#             self.error_occurred.emit(f"分割过程中发生错误: {traceback.format_exc()}")
#         finally:
#             self.segmentation_finished.emit()
#             self.status_update.emit("分割过程已完成或终止。")
#
#
# class VideoSegmentationApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("息肉视频分割应用")  # Window title 窗口标题
#         self.setGeometry(100, 100, 1200, 800)  # Initial window size (初始窗口大小)
#
#         self.worker = None  # Initialize worker (初始化工作线程)
#         self.init_ui()
#
#         # Default paths (默认路径)
#         # Ensure these default paths are created if running for the first time
#         self.current_cfg_path = os.path.join("configs", "New_PolypVideoDataset.yaml")
#         self.current_checkpoint_path = os.path.join("checkpoints", "dummy", "New_PolypVideoDataset_15.ckpt")
#         self.current_output_dir = "./pred_output1"
#
#         self.update_path_labels()
#
#         # Apply some styling (应用样式)
#         self.setStyleSheet("""
#             QMainWindow {
#                 background-color: #2e2e2e;
#                 color: #e0e0e0;
#             }
#             QPushButton {
#                 background-color: #4CAF50;
#                 color: white;
#                 border-radius: 5px;
#                 padding: 8px 15px;
#                 font-weight: bold;
#             }
#             QPushButton:hover {
#                 background-color: #45a049;
#             }
#             QLineEdit {
#                 background-color: #3e3e3e;
#                 border: 1px solid #555;
#                 border-radius: 3px;
#                 padding: 5px;
#                 color: #e0e0e0;
#             }
#             QLabel {
#                 color: #e0e0e0;
#             }
#             QCheckBox {
#                 color: #e0e0e0;
#             }
#             QProgressBar {
#                 text-align: center;
#                 color: black;
#                 background-color: #444;
#                 border-radius: 5px;
#             }
#             QProgressBar::chunk {
#                 background-color: #007bff;
#                 border-radius: 5px;
#             }
#             QTabWidget::pane {
#                 border: 1px solid #555;
#             }
#             QTabBar::tab {
#                 background: #3e3e3e;
#                 color: #e0e0e0;
#                 padding: 8px;
#                 border-top-left-radius: 5px;
#                 border-top-right-radius: 5px;
#             }
#             QTabBar::tab:selected {
#                 background: #555;
#                 color: white;
#             }
#              /* Added style for the image labels to ensure text is visible */
#             QLabel#image_display_label { /* Using object name for specific styling */
#                 border: 1px solid #666;
#                 background-color: #3e3e3e;
#                 font-size: 16px;
#                 font-weight: bold;
#                 color: #aaaaaa; /* Lighter grey for placeholder text */
#             }
#         """)
#
#     def init_ui(self):
#         # Create central widget and main layout (创建中心部件和主布局)
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
#         main_layout = QHBoxLayout(central_widget)
#
#         # --- Control Panel (Left Side) --- (控制面板 - 左侧)
#         control_panel = QWidget()
#         control_layout = QVBoxLayout(control_panel)
#         control_panel.setFixedWidth(300)  # Fixed width for control panel (固定宽度)
#
#         # File Selection (文件选择)
#         control_layout.addWidget(QLabel("<h4>配置与文件</h4>"))  # Section Title (节标题)
#         self.cfg_path_label = QLineEdit()
#         self.cfg_path_label.setReadOnly(True)
#         btn_cfg = QPushButton("选择配置文件")  # Button text (按钮文本)
#         btn_cfg.clicked.connect(self.select_config_file)
#         control_layout.addWidget(self.cfg_path_label)
#         control_layout.addWidget(btn_cfg)
#
#         self.checkpoint_path_label = QLineEdit()
#         self.checkpoint_path_label.setReadOnly(True)
#         btn_checkpoint = QPushButton("选择模型检查点")  # Button text (按钮文本)
#         btn_checkpoint.clicked.connect(self.select_checkpoint_file)
#         control_layout.addWidget(self.checkpoint_path_label)
#         control_layout.addWidget(btn_checkpoint)
#
#         self.output_dir_label = QLineEdit()
#         self.output_dir_label.setReadOnly(True)
#         btn_output_dir = QPushButton("选择输出目录")  # Button text (按钮文本)
#         btn_output_dir.clicked.connect(self.select_output_directory)
#         control_layout.addWidget(self.output_dir_label)
#         control_layout.addWidget(btn_output_dir)
#
#         # Options (选项)
#         control_layout.addWidget(QLabel("<h4>分割选项</h4>"))  # Section Title (节标题)
#         self.checkbox_overlay = QCheckBox("生成叠加图像")  # Checkbox text (复选框文本)
#         self.checkbox_overlay.setChecked(True)
#         control_layout.addWidget(self.checkbox_overlay)
#
#         self.checkbox_no_freq_maps = QCheckBox("禁用频域图显示")  # Checkbox text (复选框文本)
#         self.checkbox_no_freq_maps.setChecked(False)  # Default to showing freq maps
#         control_layout.addWidget(self.checkbox_no_freq_maps)
#
#         # Action Buttons (操作按钮)
#         control_layout.addStretch()  # Push buttons to bottom (将按钮推到底部)
#         self.btn_start = QPushButton("开始分割")  # Button text (按钮文本)
#         self.btn_start.clicked.connect(self.start_segmentation)
#         control_layout.addWidget(self.btn_start)
#
#         self.btn_stop = QPushButton("停止分割")  # Button text (按钮文本)
#         self.btn_stop.clicked.connect(self.stop_segmentation)
#         self.btn_stop.setEnabled(False)  # Initially disabled (初始禁用)
#         control_layout.addWidget(self.btn_stop)
#
#         main_layout.addWidget(control_panel)
#
#         # --- Display Area (Right Side) --- (显示区域 - 右侧)
#         display_area = QWidget()
#         display_layout = QVBoxLayout(display_area)
#
#         # Tab Widget for different views (用于不同视图的标签页部件)
#         self.tab_widget = QTabWidget()
#         display_layout.addWidget(self.tab_widget)
#
#         # Helper function to create image display labels
#         def create_image_label(text):
#             label = QLabel(text)
#             label.setAlignment(Qt.AlignCenter)
#             label.setScaledContents(True)  # Important for scaling images
#             label.setObjectName("image_display_label")  # For specific styling
#             return label
#
#         # Original Image Tab (原始图像标签页)
#         self.orig_label = create_image_label("原始图像将在此处显示")  # Label text (标签文本)
#         self.tab_widget.addTab(self.orig_label, "原始图像")  # Tab title (标签页标题)
#
#         # Predicted Mask Tab (预测掩码标签页)
#         self.mask_label = create_image_label("预测掩码将在此处显示")  # Label text (标签文本)
#         self.tab_widget.addTab(self.mask_label, "掩码")  # Tab title (标签页标题)
#
#         # Overlay Tab (叠加图像标签页)
#         self.overlay_label = create_image_label("叠加图像将在此处显示")  # Label text (标签文本)
#         self.tab_widget.addTab(self.overlay_label, "叠加")  # Tab title (标签页标题)
#
#         # Low Frequency Tab (低频图标签页)
#         self.low_freq_label = create_image_label("低频图将在此处显示")  # Label text (标签文本)
#         self.tab_widget.addTab(self.low_freq_label, "低频")  # Tab title (标签页标题)
#
#         # High Frequency Tab (高频图标签页)
#         self.high_freq_label = create_image_label("高频图将在此处显示")  # Label text (标签文本)
#         self.tab_widget.addTab(self.high_freq_label, "高频")  # Tab title (标签页标题)
#
#         # Progress Bar and Status Bar (进度条和状态栏)
#         self.progress_bar = QProgressBar()
#         self.progress_bar.setValue(0)
#         display_layout.addWidget(self.progress_bar)
#
#         self.status_label = QLabel("准备就绪。请选择文件并点击 '开始分割'。")  # Initial status text (初始状态文本)
#         self.statusBar().addWidget(self.status_label)
#
#         main_layout.addWidget(display_area)
#
#     def update_path_labels(self):
#         # Update path labels (更新路径标签)
#         self.cfg_path_label.setText(self.current_cfg_path)
#         self.checkpoint_path_label.setText(self.current_checkpoint_path)
#         self.output_dir_label.setText(self.current_output_dir)
#
#     @pyqtSlot()
#     def select_config_file(self):
#         # Open file dialog for config file (打开配置文件对话框)
#         file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "YAML 文件 (*.yaml);;所有文件 (*)")
#         if file_path:
#             self.current_cfg_path = file_path
#             self.cfg_path_label.setText(file_path)
#
#     @pyqtSlot()
#     def select_checkpoint_file(self):
#         # Open file dialog for checkpoint file (打开检查点文件对话框)
#         file_path, _ = QFileDialog.getOpenFileName(self, "选择模型检查点", "", "检查点文件 (*.ckpt);;所有文件 (*)")
#         if file_path:
#             self.current_checkpoint_path = file_path
#             self.checkpoint_path_label.setText(file_path)
#
#     @pyqtSlot()
#     def select_output_directory(self):
#         # Open directory dialog for output directory (打开输出目录对话框)
#         dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
#         if dir_path:
#             self.current_output_dir = dir_path
#             self.output_dir_label.setText(dir_path)
#
#     @pyqtSlot()
#     def start_segmentation(self):
#         # Start segmentation process (开始分割过程)
#         if self.worker and self.worker.isRunning():
#             QMessageBox.warning(self, "警告", "分割正在运行中。")  # Warning message (警告信息)
#             return
#
#         # Clear previous results (清除之前的结果) or set placeholder text
#         self.orig_label.setText("正在加载原始图像...")
#         self.mask_label.setText("正在加载预测掩码...")
#         self.overlay_label.setText("正在加载叠加图像...")
#         self.low_freq_label.setText("正在加载低频图...")
#         self.high_freq_label.setText("正在加载高频图...")
#
#         self.orig_label.clear()  # Clear actual pixmap
#         self.mask_label.clear()
#         self.overlay_label.clear()
#         self.low_freq_label.clear()
#         self.high_freq_label.clear()
#
#         self.progress_bar.setValue(0)
#         self.status_label.setText("正在开始分割...")  # Status update (状态更新)
#
#         self.btn_start.setEnabled(False)
#         self.btn_stop.setEnabled(True)
#
#         self.worker = SegmentationWorker(
#             self.current_cfg_path,
#             self.current_checkpoint_path,
#             self.current_output_dir,
#             self.checkbox_overlay.isChecked(),
#             self.checkbox_no_freq_maps.isChecked()
#         )
#         # Connect signals to slots (连接信号与槽)
#         self.worker.frame_processed.connect(self.update_display)
#         self.worker.segmentation_finished.connect(self.segmentation_finished)
#         self.worker.progress_update.connect(self.progress_bar.setValue)
#         self.worker.status_update.connect(self.status_label.setText)
#         self.worker.error_occurred.connect(self.show_error_message)
#
#         self.worker.start()  # Start the worker thread (启动工作线程)
#
#     @pyqtSlot()
#     def stop_segmentation(self):
#         # Stop segmentation process (停止分割过程)
#         if self.worker and self.worker.isRunning():
#             self.worker.stop()
#             self.btn_stop.setEnabled(False)  # Will be re-enabled by finished signal (完成后重新启用)
#             self.status_label.setText("正在停止分割，请稍候...")  # Status update (状态更新)
#         else:
#             QMessageBox.information(self, "信息", "没有正在运行的分割任务。")  # Info message (信息提示)
#
#     @pyqtSlot(QPixmap, QPixmap, QPixmap, QPixmap, QPixmap, int, int)
#     def update_display(self, orig_qpix, mask_qpix, overlay_qpix, low_freq_qpix, high_freq_qpix, current_frame,
#                        total_frames):
#         # Update display labels with new pixmaps (使用新的图像更新显示标签)
#         # Only set pixmap if it's not null, otherwise clear the label
#         self._set_pixmap_or_clear(self.orig_label, orig_qpix, "原始图像")
#         self._set_pixmap_or_clear(self.mask_label, mask_qpix, "预测掩码")
#
#         if self.checkbox_overlay.isChecked():
#             self._set_pixmap_or_clear(self.overlay_label, overlay_qpix, "叠加图像")
#         else:
#             self.overlay_label.clear()  # Clear if overlay is not requested (如果未请求叠加，则清除)
#             self.overlay_label.setText("已禁用叠加图像显示")  # Placeholder text
#
#         if not self.checkbox_no_freq_maps.isChecked():
#             self._set_pixmap_or_clear(self.low_freq_label, low_freq_qpix, "低频图")
#             self._set_pixmap_or_clear(self.high_freq_label, high_freq_qpix, "高频图")
#         else:
#             self.low_freq_label.clear()
#             self.high_freq_label.clear()
#             self.low_freq_label.setText("已禁用低频图显示")  # Placeholder text
#             self.high_freq_label.setText("已禁用高频图显示")  # Placeholder text
#
#         self.status_label.setText(f"正在处理帧 {current_frame}/{total_frames}")  # Status update (状态更新)
#
#     def _set_pixmap_or_clear(self, label, pixmap, placeholder_text=""):
#         """Helper to set pixmap or clear and set placeholder text if pixmap is null."""
#         if not pixmap.isNull():
#             label.setPixmap(pixmap)
#             label.setText("")  # Clear placeholder text if image is set
#         else:
#             label.clear()
#             label.setText(f"无法显示 {placeholder_text}\n(可能无数据或加载失败)")
#
#     @pyqtSlot()
#     def segmentation_finished(self):
#         # Handle segmentation finished (处理分割完成事件)
#         self.status_label.setText("分割过程已完成。")  # status (最终状态)
#         self.btn_start.setEnabled(True)
#         self.btn_stop.setEnabled(False)
#         self.progress_bar.setValue(100)
#         if self.worker:
#             self.worker.quit()  # Safely quit the thread (安全退出线程)
#             self.worker.wait()  # Wait for it to finish (等待线程结束)
#
#     @pyqtSlot(str)
#     def show_error_message(self, message):
#         # Show error message box (显示错误消息框)
#         QMessageBox.critical(self, "错误", message)  # Error title and message (错误标题和信息)
#         self.status_label.setText("发生错误。")  # Status update (状态更新)
#         self.btn_start.setEnabled(True)
#         self.btn_stop.setEnabled(False)
#         self.progress_bar.setValue(0)
#
#         # Clear images and show error text on labels
#         self.orig_label.clear()
#         self.mask_label.clear()
#         self.overlay_label.clear()
#         self.low_freq_label.clear()
#         self.high_freq_label.clear()
#
#         self.orig_label.setText("错误：无法显示图像\n" + message)
#         self.mask_label.setText("错误：无法显示图像\n" + message)
#         self.overlay_label.setText("错误：无法显示图像\n" + message)
#         self.low_freq_label.setText("错误：无法显示图像\n" + message)
#         self.high_freq_label.setText("错误：无法显示图像\n" + message)
#
#         if self.worker:
#             self.worker.quit()
#             self.worker.wait()
#
#     def closeEvent(self, event):
#         # Handle close event (处理关闭事件)
#         if self.worker and self.worker.isRunning():
#             reply = QMessageBox.question(self, '确认退出',  # Confirmation title (确认标题)
#                                          "分割仍在运行中。您确定要退出吗？",  # Confirmation message (确认信息)
#                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
#             if reply == QMessageBox.Yes:
#                 self.worker.stop()
#                 self.worker.wait(5000)  # Wait up to 5 seconds for thread to finish (最多等待5秒)
#                 if self.worker.isRunning():
#                     print("工作线程未终止，强制退出。")  # Force exit message (强制退出信息)
#                 event.accept()
#             else:
#                 event.ignore()
#         else:
#             event.accept()
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = VideoSegmentationApp()
#     window.show()
#     sys.exit(app.exec_())



### Why the DummyTrainer Needs Dataset Access (and the fix)

# The `DummyTrainer`'s `predict` method needs access to the dataset's properties, particularly the `CLASSES` attribute (e.g., "background", "polyp", "instrument"), which is part of your `val_set`.
#
# Instead of trying to modify the `DataLoader` directly, you should pass the `val_set` (which is your dataset) directly to the `DummyTrainer` or the `DummyModel` if it needs to access `CLASSES` during its `predict_step`.
#
# Given that your `DummyModel` currently takes `cfg` as an argument, and `cfg` is where `DATASET` information (including class names if they were truly in the config) would reside, the cleanest solution is to pass the `CLASSES` list from `val_set` to your `DummyArgs` object, which then passes it to `build_model` and subsequently to `DummyModel`.
#
# -----
#
# ### Solution: Pass `class_list` via `DummyArgs`
#
# I'll modify the `SegmentationWorker` to pass `temp_args.class_list` to the `build_model` function, which then can be used by the `DummyModel`. This avoids modifying the `DataLoader` after initialization.
#
# Here's the updated `SegmentationWorker` class with the fix:
#
# ```python

### **完整的 `pred.py` 代码 (包含GUI和分割逻辑)**

# 您可以直接复制粘贴到您的
# `.py
# ` 文件中。
#
# ** 再次提醒：在运行此代码之前，请务必仔细阅读下方的“重要前置条件”和“使用说明”部分。 ** 这些是确保程序正常运行的关键。
#
# ```python
# 1. ** 用户选择一张原始图片。 **
# 2. ** 程序根据这张原始图片，立即加载其对应的“预测好的”结果图像（掩码、叠加图、频域图等）。 **
# 3. ** 将这些加载的图片显示在
# GUI
# 上，而不需要实际运行模型的预测流程。 **
#
# 你目前的程序在“预测模式”下是生成和处理 ** 整个视频流 ** 的模拟数据（虽然只有几帧），并保存到指定的输出路径。如果你想直接从这些“预测好的路径”加载图片显示，那我们需要增加一个\ * \ * “查看结果”或“加载图像”模式\ * \ * ，它独立于当前的“开始分割”按钮。
#
# -----
#
# ### 实现“加载并显示单个图像及其预测结果”功能
#
# 为了实现这个功能，我将对现有代码进行以下修改：
#
# 1. ** 添加一个新的按钮： ** “加载结果图像”。
# 2. ** 添加一个新的槽函数： ** `load_and_display_result_image()`，当点击新按钮时调用。
# 3. ** 修改
# `load_and_display_result_image()`
# 逻辑： **
# *通过
# `QFileDialog`
# 让用户选择一个 ** 原始图像文件 **。
# *根据选定的原始图像路径， ** 推断 ** 其对应的预测结果文件（掩码、叠加图、低频图、高频图）的路径。这需要你了解你的预测结果是如何命名和存储的。
# *加载这些推断出来的预测结果图像。
# *将这些图像转换成
# `QPixmap`
# 并显示到
# GUI
# 对应的
# QLabel
# 上。
#
# -----
#
# ### 修改思路（基于你提供的代码结构）
#
# 由于你目前的
# `SegmentationWorker`
# 是在“模拟”生成预测结果和保存它们，这意味着你保存的路径结构是：
# `self.output_pred_path` / `video_subfolder` / `[not_overlay | overlay | low_freq | high_freq]` / `[base_name]
# _pred.png
# ` 等。
#
# 所以，当用户选择一个原始图片
# `path / to / your / image / video_X / frame_YYYY.jpg`
# 时，我们需要知道
# `base_name`(例如
# `frame_YYYY`) 和
# `video_subfolder`(例如
# `video_X`)，然后据此构建预测结果的路径。
#
# 例如，如果原始图片是
# `dummy_data / video_001 / frame_0000.jpg`，且你的
# `current_output_dir`
# 是
# `. / pred2
# `，那么预测结果的路径可能是：
#
# *掩码：`. / pred2 / DummyPolypDataset / video_001 / not_overlay / frame_0000_pred.png
# `
# *叠加：`. / pred2 / DummyPolypDataset / video_001 / overlay / frame_0000_overlay.png
# `
# *低频：`. / pred2 / DummyPolypDataset / video_001 / low_freq / frame_0000_low.png
# `
# *高频：`. / pred2 / DummyPolypDataset / video_001 / high_freq / frame_0000_high.png
# `
#
# ** 重要前提： ** 这种直接加载的前提是 ** 你已经运行过一次分割，并且这些预测结果文件已经实际保存到了磁盘上。 ** 如果你没有运行过分割或者你的模拟器没有实际保存文件，那么这个加载功能将无法找到文件。你目前的模拟器是会保存文件的，所以这个功能是可行的。
#
# -----
#
# ### 修改后的代码
#
# 我将直接在你的代码上进行修改，添加这个新功能。
#
# ```python
# import sys
# import os
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#     QPushButton, QLabel, QFileDialog, QLineEdit, QCheckBox, QProgressBar,
#     QMessageBox, QTabWidget, QSplitter, QGroupBox
# )
# from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
#
# import traceback
# import yaml
# import warnings
#
# warnings.filterwarnings("ignore")
#
# # --- Global Flag for PyTorch/Lightning Availability ---
# _TORCH_AVAILABLE = False
# try:
#     import torch
#     import pytorch_lightning as pl
#
#     _TORCH_AVAILABLE = True
#     print("PyTorch and PyTorch Lightning detected, but running prediction in dummy simulation mode for GUI.")
# except ImportError:
#     print("PyTorch or PyTorch Lightning not found. Running in full dummy simulation mode.")
#
#
#     class DummyPytorchLightningModule:
#         def predict_step(self, batch, batch_idx):
#             pass
#
#
#     pl = type('pl_module', (), {'LightningModule': DummyPytorchLightningModule})()
#
#
#     class DummyTorch:
#         def from_numpy(self, x): return x
#
#         def save(self, obj, f): pass
#
#         def Tensor(self, *args, **kwargs): return object()
#
#         @property
#         def cuda(self):
#             class DummyCuda:
#                 def is_available(self): return False
#
#             return DummyCuda()
#
#
#     torch = DummyTorch()
#
#
# # --- Mock/Dummy Implementations for the GUI to function without real models ---
#
# class DummyTrainer:
#     def __init__(self, devices, accelerator, logger=False, enable_model_summary=False, enable_progress_bar=False):
#         print(f"DummyTrainer: Initialized (always simulating, ignoring real device/accelerator settings).")
#
#     def predict(self, model, dataloaders, ckpt_path=None):
#         print(f"DummyTrainer: Simulating prediction. (Checkpoint path: '{ckpt_path}' will NOT be loaded).")
#         all_predictions = []
#         for batch_idx, batch in enumerate(dataloaders):
#             prediction_output = model.predict_step(batch, batch_idx)
#             all_predictions.append(prediction_output)
#         return all_predictions
#
#
# class DummyDataset:
#     CLASSES = ["background", "polyp", "instrument"]
#     frames_per_video = 100
#
#     def __len__(self):
#         return 5
#
#     def __getitem__(self, idx):
#         return idx
#
#
# def create_dummy_data_loader(cfg, dataset_instance):
#     clip_len_pred = cfg["DATASET"].get("clip_len_pred", 8)
#     stride_pred = cfg["DATASET"].get("stride_pred", clip_len_pred)
#     batch_size = cfg["PREDICT"].get("batch_size", 1)
#
#     if not _TORCH_AVAILABLE:
#         class MockDataLoader:
#             def __init__(self, dataset, sampler, batch_size, num_workers, pin_memory):
#                 self._len = len(sampler)
#                 print(f"MockDataLoader: Initialized with {self._len} batches.")
#
#             def __len__(self): return self._len
#
#             def __iter__(self):
#                 for i in range(self._len):
#                     yield i
#
#         DataLoader = MockDataLoader
#     else:
#         from torch.utils.data import DataLoader
#
#     sampler = SlidingWindowClipSampler(
#         dataset=dataset_instance,
#         clip_len=clip_len_pred,
#         stride=stride_pred,
#         shuffle=False,
#         drop_last=False
#     )
#     num_clips = len(sampler)
#     print(
#         f"SlidingWindowClipSampler: Initialized with {dataset_instance.frames_per_video * len(dataset_instance)} total frames, {num_clips} conceptual clips.")
#
#     dummy_loader = DataLoader(dataset_instance, sampler=sampler, batch_size=batch_size, num_workers=0, pin_memory=False)
#     return dummy_loader
#
#
# def build_model(args, cfg):
#     print("Building dummy model (always generating numpy arrays)...")
#
#     class DummyModel(pl.LightningModule):
#         def __init__(self, cfg_for_model, class_list):
#             super().__init__()
#             self.cfg = cfg_for_model
#             self.class_list = class_list
#
#         def predict_step(self, batch, batch_idx):
#             frames_in_clip = self.cfg["DATASET"].get("clip_len_pred", 8)
#
#             dummy_preds = np.random.randint(0, len(self.class_list), (frames_in_clip, 256, 256), dtype=np.uint8)
#
#             dummy_img_paths = []
#             dummy_video_dir = os.path.join("dummy_data",
#                                            f"video_{batch_idx + 1:03d}")  # Ensure unique video dirs for saving
#             os.makedirs(dummy_video_dir, exist_ok=True)
#
#             for i in range(frames_in_clip):
#                 frame_idx_in_sequence = batch_idx * frames_in_clip + i
#                 path = os.path.join(dummy_video_dir, f"frame_{frame_idx_in_sequence:04d}.jpg")
#                 if not os.path.exists(path):
#                     dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
#                     color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
#                     cv2.rectangle(dummy_img, (50, 50), (200, 200), color, -1)
#                     cv2.putText(dummy_img, f"Clip {batch_idx} Frame {i}", (20, 40),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                     cv2.imwrite(path, dummy_img)
#                 dummy_img_paths.append(path)
#
#             dummy_low_freqs = np.random.rand(frames_in_clip, 1, 128, 128).astype(np.float32)
#             dummy_high_freqs = np.random.rand(frames_in_clip, 1, 128, 128).astype(np.float32)
#
#             return {
#                 "preds": dummy_preds,
#                 "img_paths": dummy_img_paths,
#                 "low_freqs": dummy_low_freqs,
#                 "high_freqs": dummy_high_freqs
#             }
#
#     return DummyModel(cfg, args.class_list)
#
#
# class SlidingWindowClipSampler:
#     def __init__(self, dataset, clip_len, stride, shuffle, drop_last):
#         self.dataset = dataset
#         self.clip_len = clip_len
#         self.stride = stride
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#
#         total_frames = getattr(dataset, 'frames_per_video', 100) * len(dataset)
#
#         if total_frames < clip_len:
#             self._length = 0
#         else:
#             self._length = (total_frames - clip_len) // stride + 1
#             if (total_frames - clip_len) % stride != 0:
#                 self._length += 1
#             if self._length < 0:
#                 self._length = 0
#
#     def __iter__(self):
#         for i in range(len(self)):
#             yield i
#
#     def __len__(self):
#         return self._length
#
#
# # --- End Mock/Dummy Implementations ---
#
#
# def convert_cv_to_qpixmap(cv_img, target_size=None):
#     if cv_img is None or cv_img.size == 0:
#         return QPixmap()
#
#     h, w = cv_img.shape[:2]
#
#     if len(cv_img.shape) == 2:
#         bytes_per_line = w
#         if cv_img.dtype != np.uint8:
#             cv_img = np.clip(cv_img, 0, 255).astype(np.uint8)
#         q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
#     else:
#         ch = cv_img.shape[2]
#         if ch == 1:
#             if cv_img.dtype != np.uint8:
#                 cv_img = np.clip(cv_img, 0, 255).astype(np.uint8)
#             bytes_per_line = w
#             q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
#         else:
#             bytes_per_line = ch * w
#             q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
#
#     if target_size:
#         return QPixmap.fromImage(q_img).scaled(target_size[0], target_size[1], Qt.KeepAspectRatio,
#                                                Qt.SmoothTransformation)
#     return QPixmap.fromImage(q_img)
#
#
# class SegmentationWorker(QThread):
#     frame_processed = pyqtSignal(QPixmap, QPixmap, QPixmap, QPixmap, QPixmap, int, int)
#     segmentation_finished = pyqtSignal()
#     progress_update = pyqtSignal(int)
#     status_update = pyqtSignal(str)
#     error_occurred = pyqtSignal(str)
#
#     def __init__(self, config_path, checkpoint_path, output_pred_path,
#                  overlay_enabled, no_freq_maps_enabled, parent=None):
#         super().__init__(parent)
#         self.config_path = config_path
#         self.checkpoint_path = checkpoint_path
#         self.output_pred_path = output_pred_path
#         self.overlay_enabled = overlay_enabled
#         self.no_freq_maps_enabled = no_freq_maps_enabled
#         self._is_running = True
#
#         self.cfg = None
#         self.model = None
#         self.val_set = None
#         self.data_loader_val = None
#         self.color_map = {
#             0: (0, 0, 0),  # background Black (RGB)
#             1: (0, 255, 0),  # Polyp Green (RGB)
#             2: (0, 0, 255),  # instrument Blue (RGB)
#         }
#
#     def stop(self):
#         self._is_running = False
#
#     def run(self):
#         self.status_update.emit("正在初始化分割工作器...")
#         try:
#             # 1. Load Configuration
#             if not os.path.exists(self.config_path):
#                 os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
#                 dummy_cfg_content = """
# DATASET:
#   dataset: "DummyPolypDataset"
#   Low_mean: [0.5]
#   Low_std: [0.5]
#   High_mean: [0.5]
#   High_std: [0.5]
#   clip_len_pred: 8
#   stride_pred: 8
#   frames_per_video: 100
# PREDICT:
#   batch_size: 1
# TRAIN:
#   num_workers: 0
#                 """
#                 with open(self.config_path, "w") as f:
#                     f.write(dummy_cfg_content)
#                 self.status_update.emit(f"警告: 配置文件未找到，已创建虚拟配置文件: {self.config_path}")
#
#             with open(self.config_path) as f:
#                 self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
#             self.status_update.emit("配置加载成功。")
#
#             if "PREDICT" not in self.cfg:
#                 self.cfg["PREDICT"] = {"batch_size": 1}
#                 self.status_update.emit("警告: 配置文件缺少 'PREDICT' 部分，已添加默认值。")
#
#             # 2. Build Dataset and DataLoader (Always use dummy implementations)
#             class DummyArgs:
#                 def __init__(self, cfg_data, checkpoint_path_val, output_pred_path_val, overlay_enabled_val,
#                              no_freq_maps_enabled_val):
#                     self.cfg = cfg_data
#                     self.checkpoint = checkpoint_path_val
#                     self.output_pred = output_pred_path_val
#                     self.overlay = overlay_enabled_val
#                     self.no_freq_maps = no_freq_maps_enabled_val
#                     self.class_list = None
#
#             temp_args = DummyArgs(self.cfg, self.checkpoint_path, self.output_pred_path, self.overlay_enabled,
#                                   self.no_freq_maps_enabled)
#
#             self.status_update.emit("正在构建验证数据集 (模拟)...")
#             self.val_set = DummyDataset()
#             if not self.val_set or len(self.val_set) == 0:
#                 raise ValueError("验证数据集为空或构建失败。")
#             temp_args.class_list = self.val_set.CLASSES
#             self.status_update.emit(f"验证数据集构建完成。模拟视频/分段数量: {len(self.val_set)}")
#
#             self.data_loader_val = create_dummy_data_loader(self.cfg, self.val_set)
#
#             if len(self.data_loader_val) == 0:
#                 raise ValueError("数据加载器为空。无法进行预测。")
#             self.status_update.emit(f"数据加载器已创建。批次（剪辑）数量: {len(self.data_loader_val)}")
#
#             # 3. Build Model (Always use DummyModel)
#             self.status_update.emit("正在构建模型 (模拟)...")
#             self.model = build_model(temp_args, self.cfg)
#             self.status_update.emit("模型构建成功。")
#
#             # 4. Setup Trainer (Always use DummyTrainer)
#             self.status_update.emit("正在设置模拟 Trainer...")
#             trainer = DummyTrainer(
#                 devices=1,
#                 accelerator="cpu",
#                 logger=False,
#                 enable_model_summary=False,
#                 enable_progress_bar=False
#             )
#             self.status_update.emit("模拟 Trainer 已就绪。")
#
#             # 5. Perform Prediction (using DummyTrainer, will not load real checkpoint)
#             if not os.path.exists(self.checkpoint_path):
#                 os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
#                 with open(self.checkpoint_path, "w") as f:
#                     f.write("DUMMY_CHECKPOINT_CONTENT_IGNORED_BY_DUMMY_TRAINER")
#                 self.status_update.emit(
#                     f"警告: 检查点文件未找到，已创建虚拟检查点占位符: {self.checkpoint_path}。此文件内容不影响模拟。")
#
#             self.status_update.emit(f"\n--- 正在进行预测 (模拟模式) ---")
#             predictions = trainer.predict(self.model, dataloaders=self.data_loader_val, ckpt_path=self.checkpoint_path)
#
#             if predictions is None:
#                 raise RuntimeError("模拟预测结果为 None。")
#             elif not isinstance(predictions, list):
#                 raise TypeError(f"模拟预测返回类型 {type(predictions)}, 预期为列表。")
#             elif len(predictions) == 0:
#                 self.error_occurred.emit("警告: 模拟预测返回空列表。没有生成预测结果。")
#                 self.segmentation_finished.emit()
#                 return
#             else:
#                 self.status_update.emit(f"\n模拟预测完成。获得 {len(predictions)} 个预测项（剪辑）。")
#
#             # 6. Process and Save Predictions
#             self.status_update.emit("--- 正在处理并显示结果 ---")
#
#             total_frames_to_process = sum([item["preds"].shape[0] for item in predictions if "preds" in item])
#             if total_frames_to_process == 0:
#                 self.status_update.emit("没有要处理的帧。")
#                 self.segmentation_finished.emit()
#                 return
#
#             processed_frames_count = 0
#             # Ensure the base output directory for this dataset is created
#             dataset_name = self.cfg["DATASET"].get("dataset", "unknown_dataset")
#             save_base_path = os.path.join(self.output_pred_path, dataset_name)
#             os.makedirs(save_base_path, exist_ok=True)
#
#             for item_idx, item in enumerate(predictions):
#                 if not self._is_running:
#                     self.status_update.emit("用户已停止分割。")
#                     break
#
#                 if not isinstance(item, dict) or not all(
#                         k in item for k in ["preds", "img_paths", "low_freqs", "high_freqs"]):
#                     self.error_occurred.emit(f"警告 [剪辑 {item_idx}]: 意外的项目格式。跳过。")
#                     skipped_frames_in_item = item.get("preds", np.empty((0, 0, 0))).shape[0]
#                     if not skipped_frames_in_item:
#                         skipped_frames_in_item = self.cfg["DATASET"].get("clip_len_pred", 8)
#                     processed_frames_count += skipped_frames_in_item
#                     percentage = int((processed_frames_count / total_frames_to_process) * 100)
#                     if percentage > 100: percentage = 100
#                     self.progress_update.emit(percentage)
#                     continue
#
#                 try:
#                     num_frames = item["preds"].shape[0]
#                     all_preds = item["preds"].astype(np.uint8)
#                     all_img_paths = item["img_paths"]
#                     all_low_freqs = item["low_freqs"]
#                     all_high_freqs = item["high_freqs"]
#                 except Exception as e:
#                     self.error_occurred.emit(f"错误: 从项目 {item_idx} 解包数据失败: {e}。跳过剪辑。")
#                     skipped_frames_in_item = self.cfg["DATASET"].get("clip_len_pred", 8)
#                     processed_frames_count += skipped_frames_in_item
#                     percentage = int((processed_frames_count / total_frames_to_process) * 100)
#                     if percentage > 100: percentage = 100
#                     self.progress_update.emit(percentage)
#                     continue
#
#                 if not (len(all_img_paths) == num_frames and all_low_freqs.shape[0] == num_frames and
#                         all_high_freqs.shape[0] == num_frames):
#                     self.error_occurred.emit(f"警告 [剪辑 {item_idx}]: 数据长度不匹配。跳过剪辑。")
#                     processed_frames_count += num_frames
#                     percentage = int((processed_frames_count / total_frames_to_process) * 100)
#                     if percentage > 100: percentage = 100
#                     self.progress_update.emit(percentage)
#                     continue
#
#                 self.status_update.emit(f"正在处理剪辑 {item_idx + 1}/{len(predictions)}, 共 {num_frames} 帧...")
#
#                 for i in range(num_frames):
#                     if not self._is_running:
#                         self.status_update.emit("用户已停止分割。")
#                         break
#
#                     pred = all_preds[i]
#                     img_path = all_img_paths[i]
#
#                     if isinstance(img_path, list):
#                         if len(img_path) == 1 and isinstance(img_path[0], str):
#                             img_path = img_path[0]
#                         else:
#                             self.error_occurred.emit(
#                                 f"警告 [剪辑 {item_idx}, 帧 {i}]: img_path 是列表但不是单个字符串: {img_path}。跳过。")
#                             processed_frames_count += 1
#                             percentage = int((processed_frames_count / total_frames_to_process) * 100)
#                             if percentage > 100: percentage = 100
#                             self.progress_update.emit(percentage)
#                             continue
#                     elif not isinstance(img_path, str):
#                         self.error_occurred.emit(
#                             f"警告 [剪辑 {item_idx}, 帧 {i}]: img_path 不是字符串: {img_path}。跳过。")
#                         processed_frames_count += 1
#                         percentage = int((processed_frames_count / total_frames_to_process) * 100)
#                         if percentage > 100: percentage = 100
#                         self.progress_update.emit(percentage)
#                         continue
#
#                     img_path = os.path.normpath(img_path)
#
#                     orig_img = cv2.imread(img_path)
#                     if orig_img is None:
#                         self.error_occurred.emit(
#                             f"警告 [剪辑 {item_idx}, 帧 {i}]: 无法加载图像: {img_path}。请检查文件是否存在和损坏。跳过。")
#                         processed_frames_count += 1
#                         percentage = int((processed_frames_count / total_frames_to_process) * 100)
#                         if percentage > 100: percentage = 100
#                         self.progress_update.emit(percentage)
#                         continue
#
#                     pred_h, pred_w = pred.shape
#
#                     color_mask_bgr = np.zeros((pred_h, pred_w, 3), dtype=np.uint8)
#                     for class_id, color_rgb in self.color_map.items():
#                         color_bgr = tuple(reversed(color_rgb))
#                         color_mask_bgr[pred == class_id] = color_bgr
#
#                     display_w, display_h = 300, 300
#
#                     orig_qpix = convert_cv_to_qpixmap(orig_img, target_size=(display_w, display_h))
#                     mask_qpix = convert_cv_to_qpixmap(color_mask_bgr, target_size=(display_w, display_h))
#
#                     overlay_qpix = QPixmap()
#                     if self.overlay_enabled:
#                         orig_img_resized = cv2.resize(orig_img, (pred_w, pred_h))
#                         overlay_img = cv2.addWeighted(orig_img_resized, 0.6, color_mask_bgr, 0.4, 0)
#                         overlay_qpix = convert_cv_to_qpixmap(overlay_img, target_size=(display_w, display_h))
#
#                     low_freq_qpix = QPixmap()
#                     high_freq_qpix = QPixmap()
#                     if not self.no_freq_maps_enabled:
#                         mean_low = float(self.cfg["DATASET"]["Low_mean"][0]) if isinstance(
#                             self.cfg["DATASET"]["Low_mean"], (list, tuple)) else float(self.cfg["DATASET"]["Low_mean"])
#                         std_low = float(self.cfg["DATASET"]["Low_std"][0]) if isinstance(self.cfg["DATASET"]["Low_std"],
#                                                                                          (list, tuple)) else float(
#                             self.cfg["DATASET"]["Low_std"])
#
#                         low_freq_data = all_low_freqs[i].squeeze(0) if all_low_freqs[i].ndim > 2 else all_low_freqs[i]
#                         denorm_low_freq = (low_freq_data * std_low + mean_low) * 255.0
#                         low_freq_resized = cv2.resize(denorm_low_freq, (pred_w, pred_h))
#                         low_freq_resized = np.clip(low_freq_resized, 0, 255).astype(np.uint8)
#                         low_freq_qpix = convert_cv_to_qpixmap(low_freq_resized, target_size=(display_w, display_h))
#
#                         mean_high = float(self.cfg["DATASET"]["High_mean"][0]) if isinstance(
#                             self.cfg["DATASET"]["High_mean"], (list, tuple)) else float(
#                             self.cfg["DATASET"]["High_mean"])
#                         std_high = float(self.cfg["DATASET"]["High_std"][0]) if isinstance(
#                             self.cfg["DATASET"]["High_std"], (list, tuple)) else float(self.cfg["DATASET"]["High_std"])
#
#                         high_freq_data = all_high_freqs[i].squeeze(0) if all_high_freqs[i].ndim > 2 else all_high_freqs[
#                             i]
#                         denorm_high_freq = (high_freq_data * std_high + mean_high) * 255.0
#                         high_freq_resized = cv2.resize(denorm_high_freq, (pred_w, pred_h))
#                         high_freq_resized = np.clip(high_freq_resized, 0, 255).astype(np.uint8)
#                         high_freq_qpix = convert_cv_to_qpixmap(high_freq_resized, target_size=(display_w, display_h))
#
#                     processed_frames_count += 1
#                     percentage = int((processed_frames_count / total_frames_to_process) * 100)
#                     if percentage > 100: percentage = 100
#
#                     self.frame_processed.emit(
#                         orig_qpix, mask_qpix, overlay_qpix, low_freq_qpix, high_freq_qpix,
#                         processed_frames_count, total_frames_to_process
#                     )
#                     self.progress_update.emit(percentage)
#
#                     # Determine video_subfolder from img_path
#                     parts = img_path.split(os.sep)
#                     video_subfolder = "unknown_video"
#                     for part in parts:  # Iterate through parts to find the 'video_XXX' segment
#                         if 'video_' in part:
#                             video_subfolder = part
#                             break
#                     if video_subfolder == "unknown_video":
#                         # Fallback for paths that don't match 'video_XXX' pattern
#                         # Try to use the grandparent directory name if img_path is like /some/dir/image.jpg
#                         # This assumes images are in subfolders directly under the dataset's video path
#                         # Example: dummy_data/video_001/frame_0000.jpg -> video_001
#                         # If the original path is simply 'frame_0000.jpg', it might default to 'unknown_video'
#                         if len(parts) >= 2:
#                             video_subfolder = parts[-2] if os.path.isdir(
#                                 os.path.join(os.path.dirname(img_path), parts[-2])) else "unknown_video"
#
#                     file_name = os.path.basename(img_path)
#                     base_name, _ = os.path.splitext(file_name)
#
#                     current_output_sub_dir = os.path.join(save_base_path, video_subfolder)
#                     os.makedirs(current_output_sub_dir, exist_ok=True)
#
#                     mask_output_dir = os.path.join(current_output_sub_dir, "not_overlay")
#                     os.makedirs(mask_output_dir, exist_ok=True)
#                     cv2.imwrite(os.path.join(mask_output_dir, f"{base_name}_pred.png"), color_mask_bgr)
#
#                     if self.overlay_enabled:
#                         orig_img_resized = cv2.resize(orig_img, (pred_w, pred_h))
#                         overlay_img_to_save = cv2.addWeighted(orig_img_resized, 0.6, color_mask_bgr, 0.4, 0)
#                         overlay_output_dir = os.path.join(current_output_sub_dir, "overlay")
#                         os.makedirs(overlay_output_dir, exist_ok=True)
#                         cv2.imwrite(os.path.join(overlay_output_dir, f"{base_name}_overlay.png"), overlay_img_to_save)
#
#                     if not self.no_freq_maps_enabled:
#                         low_freq_output_dir = os.path.join(current_output_sub_dir, "low_freq")
#                         high_freq_output_dir = os.path.join(current_output_sub_dir, "high_freq")
#                         os.makedirs(low_freq_output_dir, exist_ok=True)
#                         os.makedirs(high_freq_output_dir, exist_ok=True)
#                         cv2.imwrite(os.path.join(low_freq_output_dir, f"{base_name}_low.png"), low_freq_resized)
#                         cv2.imwrite(os.path.join(high_freq_output_dir, f"{base_name}_high.png"), high_freq_resized)
#
#         except Exception as e:
#             error_traceback = traceback.format_exc()
#             print(f"ERROR: An unhandled exception occurred in SegmentationWorker:\n{error_traceback}")
#             self.error_occurred.emit(f"分割过程中发生错误: {error_traceback}")
#         finally:
#             self.segmentation_finished.emit()
#             self.status_update.emit("分割过程已完成或终止。")
#
#
# class VideoSegmentationApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("消化内镜影像分析软件")
#         self.setGeometry(100, 100, 1200, 800)
#
#         self.worker = None
#
#         self.init_ui()
#
#         # Set the correct default paths as per your request
#         self.current_cfg_path = "configs/New_PolypVideoDataset.yaml"
#         self.current_checkpoint_path = "checkpoints/xxx/xx/New_PolypVideoDataset_15/New_PolypVideoDataset_15.ckpt"
#         self.current_output_dir = "./pred2"  # Default output directory
#
#         self.update_path_labels()
#
#         self.setStyleSheet("""
#             QMainWindow {
#                 background-color: #2e2e2e;
#                 color: #e0e0e0;
#             }
#             QWidget#ControlPanel {
#                 background-color: #383838;
#                 border-right: 1px solid #444;
#             }
#             QPushButton {
#                 background-color: #4CAF50; /* Green */
#                 color: white;
#                 border: none;
#                 border-radius: 8px;
#                 padding: 10px 20px;
#                 font-size: 14px;
#                 font-weight: bold;
#                 margin: 5px;
#             }
#             QPushButton:hover {
#                 background-color: #45a049;
#             }
#             QPushButton:pressed {
#                 background-color: #3e8e41;
#             }
#             QPushButton#StopButton {
#                 background-color: #f44336; /* Red */
#             }
#             QPushButton#StopButton:hover {
#                 background-color: #da190b;
#             }
#             QPushButton:disabled {
#                 background-color: #555;
#                 color: #aaa;
#             }
#             QLineEdit {
#                 background-color: #3e3e3e;
#                 border: 1px solid #555;
#                 border-radius: 5px;
#                 padding: 8px;
#                 color: #e0e0e0;
#                 font-size: 13px;
#             }
#             QLabel {
#                 color: #e0e0e0;
#                 font-size: 13px;
#             }
#             QLabel[text^="<h4>"] {
#                 color: #87CEEB; /* Light Blue */
#                 font-size: 16px;
#                 font-weight: bold;
#                 padding-top: 10px;
#                 padding-bottom: 5px;
#             }
#             QCheckBox {
#                 color: #e0e0e0;
#                 font-size: 13px;
#                 padding: 5px 0;
#             }
#             QProgressBar {
#                 text-align: center;
#                 color: white;
#                 background-color: #444;
#                 border-radius: 7px;
#                 height: 25px;
#                 margin: 10px 0;
#             }
#             QProgressBar::chunk {
#                 background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:1 #007bff);
#                 border-radius: 7px;
#             }
#             QTabWidget::pane {
#                 border: 1px solid #555;
#                 background-color: #2e2e2e;
#             }
#             QTabBar::tab {
#                 background: #3e3e3e;
#                 color: #e0e0e0;
#                 padding: 10px 15px;
#                 border-top-left-radius: 8px;
#                 border-top-right-radius: 8px;
#                 margin-right: 2px;
#                 font-weight: bold;
#             }
#             QTabBar::tab:selected {
#                 background: #555;
#                 color: white;
#                 border-top: 2px solid #007bff;
#             }
#             QTabBar::tab:hover {
#                 background: #4a4a4a;
#             }
#             QLabel#image_display_label {
#                 border: 1px dashed #666;
#                 background-color: #3e3e3e;
#                 font-size: 16px;
#                 font-weight: bold;
#                 color: #aaaaaa;
#                 qproperty-alignment: AlignCenter;
#             }
#             QStatusBar {
#                 background-color: #383838;
#                 color: #e0e0e0;
#                 font-size: 12px;
#                 padding: 5px;
#             }
#             QSplitter::handle {
#                 background-color: #444;
#                 width: 5px;
#             }
#             QGroupBox {
#                 border: 1px solid #555;
#                 border-radius: 5px;
#                 margin-top: 2ex; /* Space for the title */
#                 color: #e0e0e0;
#                 font-size: 14px;
#                 font-weight: bold;
#             }
#             QGroupBox::title {
#                 subcontrol-origin: margin;
#                 subcontrol-position: top left; /* Position at top left */
#                 padding: 0 3px;
#                 background-color: #383838; /* Match panel background */
#             }
#         """)
#
#     def init_ui(self):
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
#         main_layout = QHBoxLayout(central_widget)
#
#         control_panel = QWidget()
#         control_panel.setObjectName("ControlPanel")
#         control_layout = QVBoxLayout(control_panel)
#         control_layout.setAlignment(Qt.AlignTop)
#
#         # Path Configuration Group
#         path_group_box = QGroupBox("文件路径配置")
#         path_layout = QVBoxLayout(path_group_box)
#
#         path_layout.addWidget(QLabel("配置文件路径:"))
#         self.cfg_path_label = QLabel("未选择")
#         self.cfg_path_label.setWordWrap(True)
#         path_layout.addWidget(self.cfg_path_label)
#         self.btn_select_cfg = QPushButton("选择配置文件")
#         self.btn_select_cfg.clicked.connect(self.select_config_file)
#         path_layout.addWidget(self.btn_select_cfg)
#
#         path_layout.addWidget(QLabel("模型检查点路径:"))
#         self.checkpoint_path_label = QLabel("未选择")
#         self.checkpoint_path_label.setWordWrap(True)
#         path_layout.addWidget(self.checkpoint_path_label)
#         self.btn_select_checkpoint = QPushButton("选择模型检查点")
#         self.btn_select_checkpoint.clicked.connect(self.select_checkpoint_file)
#         path_layout.addWidget(self.btn_select_checkpoint)
#
#         path_layout.addWidget(QLabel("预测结果输出目录:"))
#         self.output_dir_label = QLabel("未选择")
#         self.output_dir_label.setWordWrap(True)
#         path_layout.addWidget(self.output_dir_label)
#         self.btn_select_output = QPushButton("选择输出目录")
#         self.btn_select_output.clicked.connect(self.select_output_directory)
#         path_layout.addWidget(self.btn_select_output)
#
#         control_layout.addWidget(path_group_box)
#
#         # Segmentation Options Group
#         options_group_box = QGroupBox("分割选项")
#         options_layout = QVBoxLayout(options_group_box)
#
#         self.checkbox_overlay = QCheckBox("显示叠加图像")
#         self.checkbox_overlay.setChecked(True)
#         options_layout.addWidget(self.checkbox_overlay)
#
#         self.checkbox_no_freq_maps = QCheckBox("禁用频域图显示")
#         self.checkbox_no_freq_maps.setChecked(False)
#         options_layout.addWidget(self.checkbox_no_freq_maps)
#
#         control_layout.addWidget(options_group_box)
#
#         # Actions Group
#         actions_group_box = QGroupBox("控制")
#         actions_layout = QVBoxLayout(actions_group_box)
#
#         self.btn_start = QPushButton("开始分割 (运行模型)")
#         self.btn_start.clicked.connect(self.start_segmentation)
#         actions_layout.addWidget(self.btn_start)
#
#         self.btn_stop = QPushButton("停止分割")
#         self.btn_stop.setObjectName("StopButton")
#         self.btn_stop.clicked.connect(self.stop_segmentation)
#         self.btn_stop.setEnabled(False)
#         actions_layout.addWidget(self.btn_stop)
#
#         # New button for loading pre-predicted results
#         self.btn_load_result = QPushButton("预测结果 ")
#         self.btn_load_result.clicked.connect(self.load_and_display_result_image)
#         actions_layout.addWidget(self.btn_load_result)
#
#         # MODIFIED: Button for selecting original and predicted image for comparison
#         self.btn_select_original_and_predicted = QPushButton("原始图像")
#         self.btn_select_original_and_predicted.clicked.connect(self.select_original_and_predicted_for_comparison)
#         actions_layout.addWidget(self.btn_select_original_and_predicted)
#
#         self.progress_bar = QProgressBar()
#         self.progress_bar.setValue(0)
#         actions_layout.addWidget(self.progress_bar)
#
#         control_layout.addWidget(actions_group_box)
#
#         control_layout.addStretch()  # Push everything to the top
#
#         image_display_widget = QWidget()
#         image_layout = QVBoxLayout(image_display_widget)
#         image_layout.setAlignment(Qt.AlignTop)
#
#         splitter = QSplitter(Qt.Horizontal)
#         splitter.addWidget(control_panel)
#         splitter.addWidget(image_display_widget)
#         splitter.setSizes([250, 950])
#         main_layout.addWidget(splitter)
#
#         self.image_tabs = QTabWidget()
#         image_layout.addWidget(self.image_tabs)
#
#         tab1 = QWidget()
#         tab1_layout = QHBoxLayout(tab1)
#         self.orig_label = QLabel("原始图像")
#         self.orig_label.setAlignment(Qt.AlignCenter)
#         self.orig_label.setScaledContents(True)
#         self.orig_label.setObjectName("image_display_label")
#         self.mask_label = QLabel("预测掩码")
#         self.mask_label.setAlignment(Qt.AlignCenter)
#         self.mask_label.setScaledContents(True)
#         self.mask_label.setObjectName("image_display_label")
#         tab1_layout.addWidget(self.orig_label)
#         tab1_layout.addWidget(self.mask_label)
#         self.image_tabs.addTab(tab1, "原始 & 掩码")
#
#         tab2 = QWidget()
#         tab2_layout = QHBoxLayout(tab2)
#         self.overlay_label = QLabel("叠加图像")
#         self.overlay_label.setAlignment(Qt.AlignCenter)
#         self.overlay_label.setScaledContents(True)
#         self.overlay_label.setObjectName("image_display_label")
#         self.low_freq_label = QLabel("低频图")
#         self.low_freq_label.setAlignment(Qt.AlignCenter)
#         self.low_freq_label.setScaledContents(True)
#         self.low_freq_label.setObjectName("image_display_label")
#         self.high_freq_label = QLabel("高频图")
#         self.high_freq_label.setAlignment(Qt.AlignCenter)
#         self.high_freq_label.setScaledContents(True)
#         self.high_freq_label.setObjectName("image_display_label")
#
#         tab2_layout.addWidget(self.overlay_label)
#         tab2_layout.addWidget(self.low_freq_label)
#         tab2_layout.addWidget(self.high_freq_label)
#         self.image_tabs.addTab(tab2, "叠加 & 频域图")
#
#         self.statusBar = self.statusBar()
#         self.status_label = QLabel("准备就绪")
#         self.statusBar.addWidget(self.status_label)
#
#     def update_path_labels(self):
#         self.cfg_path_label.setText(self.current_cfg_path)
#         self.checkpoint_path_label.setText(self.current_checkpoint_path)
#         self.output_dir_label.setText(self.current_output_dir)
#
#     @pyqtSlot()
#     def select_config_file(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "YAML 文件 (*.yaml);;所有文件 (*)")
#         if file_path:
#             self.current_cfg_path = file_path
#             self.cfg_path_label.setText(file_path)
#
#     @pyqtSlot()
#     def select_checkpoint_file(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "选择模型检查点", "", "检查点文件 (*.ckpt);;所有文件 (*)")
#         if file_path:
#             self.current_checkpoint_path = file_path
#             self.checkpoint_path_label.setText(file_path)
#
#     @pyqtSlot()
#     def select_output_directory(self):
#         dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录", "./")
#         if dir_path:
#             self.current_output_dir = dir_path
#             self.output_dir_label.setText(dir_path)
#
#     @pyqtSlot()
#     def start_segmentation(self):
#         if self.worker and self.worker.isRunning():
#             QMessageBox.warning(self, "警告", "分割正在运行中。")
#             return
#
#         # Try to load config for dataset name
#         cfg = {}
#         if os.path.exists(self.current_cfg_path):
#             try:
#                 with open(self.current_cfg_path) as f:
#                     cfg = yaml.load(f, Loader=yaml.SafeLoader)
#             except Exception as e:
#                 QMessageBox.critical(self, "配置错误", f"无法加载配置文件: {e}")
#                 return
#         else:
#             # If config doesn't exist, create a dummy one
#             os.makedirs(os.path.dirname(self.current_cfg_path), exist_ok=True)
#             dummy_cfg_content = """
# DATASET:
#   dataset: "DummyPolypDataset"
#   Low_mean: [0.5]
#   Low_std: [0.5]
#   High_mean: [0.5]
#   High_std: [0.5]
#   clip_len_pred: 8
#   stride_pred: 8
#   frames_per_video: 100
# PREDICT:
#   batch_size: 1
# TRAIN:
#   num_workers: 0
#             """
#             try:
#                 with open(self.current_cfg_path, "w") as f:
#                     f.write(dummy_cfg_content)
#                 QMessageBox.information(self, "配置创建",
#                                         f"配置文件不存在，已为您创建虚拟配置文件: {self.current_cfg_path}")
#                 self.cfg_path_label.setText(self.current_cfg_path)
#                 cfg = yaml.load(dummy_cfg_content, Loader=yaml.SafeLoader)  # Load newly created config
#             except Exception as e:
#                 QMessageBox.critical(self, "错误", f"无法创建虚拟配置文件: {e}")
#                 return
#
#         if not self.current_output_dir:
#             QMessageBox.warning(self, "输出目录错误", "请选择一个有效的输出目录。")
#             return
#
#         self.orig_label.clear()
#         self.mask_label.clear()
#         self.overlay_label.clear()
#         self.low_freq_label.clear()
#         self.high_freq_label.clear()
#
#         self.orig_label.setText("正在加载原始图像...")
#         self.mask_label.setText("正在加载预测掩码...")
#         self.overlay_label.setText("正在加载叠加图像...")
#         self.low_freq_label.setText("正在加载低频图...")
#         self.high_freq_label.setText("正在加载高频图...")
#
#         self.progress_bar.setValue(0)
#         self.status_label.setText("正在开始分割...")
#
#         self.btn_start.setEnabled(False)
#         self.btn_load_result.setEnabled(False)  # Disable load button during segmentation
#         self.btn_select_original_and_predicted.setEnabled(False)  # Disable new button
#         self.btn_stop.setEnabled(True)
#
#         self.worker = SegmentationWorker(
#             self.current_cfg_path,
#             self.current_checkpoint_path,
#             self.current_output_dir,
#             self.checkbox_overlay.isChecked(),
#             self.checkbox_no_freq_maps.isChecked()
#         )
#         self.worker.frame_processed.connect(self.update_display)
#         self.worker.segmentation_finished.connect(self.segmentation_finished)
#         self.worker.progress_update.connect(self.progress_bar.setValue)
#         self.worker.status_update.connect(self.status_label.setText)
#         self.worker.error_occurred.connect(self.show_error_message)
#
#         self.worker.start()
#
#     @pyqtSlot()
#     def stop_segmentation(self):
#         if self.worker and self.worker.isRunning():
#             self.worker.stop()
#             self.btn_stop.setEnabled(False)
#             self.status_label.setText("正在停止分割，请稍候...")
#         else:
#             QMessageBox.information(self, "信息", "没有正在运行的分割任务。")
#
#     @pyqtSlot(QPixmap, QPixmap, QPixmap, QPixmap, QPixmap, int, int)
#     def update_display(self, orig_qpix, mask_qpix, overlay_qpix, low_freq_qpix, high_freq_qpix, current_frame,
#                        total_frames):
#         self._set_pixmap_or_clear(self.orig_label, orig_qpix, "原始图像")
#         self._set_pixmap_or_clear(self.mask_label, mask_qpix, "预测掩码")
#
#         if self.checkbox_overlay.isChecked():
#             self._set_pixmap_or_clear(self.overlay_label, overlay_qpix, "叠加图像")
#         else:
#             self.overlay_label.clear()
#             self.overlay_label.setText("已禁用叠加图像显示")
#
#         if not self.checkbox_no_freq_maps.isChecked():
#             self._set_pixmap_or_clear(self.low_freq_label, low_freq_qpix, "低频图")
#             self._set_pixmap_or_clear(self.high_freq_label, high_freq_qpix, "高频图")
#         else:
#             self.low_freq_label.clear()
#             self.high_freq_label.clear()
#             self.low_freq_label.setText("已禁用低频图显示")
#             self.high_freq_label.setText("已禁用高频图显示")
#
#         self.status_label.setText(f"正在处理帧 {current_frame}/{total_frames}")
#
#     def _set_pixmap_or_clear(self, label, pixmap, placeholder_text=""):
#         if not pixmap.isNull():
#             label.setPixmap(pixmap)
#             label.setText("")
#         else:
#             label.clear()
#             label.setText(f"无法显示 {placeholder_text}\n(可能无数据或加载失败)")
#
#     @pyqtSlot()
#     def segmentation_finished(self):
#         self.status_label.setText("分割过程已完成。")
#         self.btn_start.setEnabled(True)
#         self.btn_load_result.setEnabled(True)
#         self.btn_select_original_and_predicted.setEnabled(True)  # Enable new button
#         self.btn_stop.setEnabled(False)
#         self.progress_bar.setValue(100)
#         if self.worker:
#             self.worker.quit()
#             self.worker.wait()
#
#     @pyqtSlot(str)
#     def show_error_message(self, message):
#         QMessageBox.critical(self, "错误", message)
#         self.status_label.setText("发生错误。")
#         self.btn_start.setEnabled(True)
#         self.btn_load_result.setEnabled(True)
#         self.btn_select_original_and_predicted.setEnabled(True)  # Enable new button
#         self.btn_stop.setEnabled(False)
#         self.progress_bar.setValue(0)
#
#         self.orig_label.clear()
#         self.mask_label.clear()
#         self.overlay_label.clear()
#         self.low_freq_label.clear()
#         self.high_freq_label.clear()
#
#         display_error = "发生错误，请查看控制台输出。"
#         if "\n" in message:
#             display_error = f"错误：{message.splitlines()[0]}..."
#
#         self.orig_label.setText(f"无法显示图像\n{display_error}")
#         self.mask_label.setText(f"无法显示图像\n{display_error}")
#         self.overlay_label.setText(f"无法显示图像\n{display_error}")
#         self.low_freq_label.setText(f"无法显示图像\n{display_error}")
#         self.high_freq_label.setText(f"无法显示图像\n{display_error}")
#
#         if self.worker:
#             self.worker.quit()
#             self.worker.wait()
#
#     @pyqtSlot()
#     def load_and_display_result_image(self):
#         if self.worker and self.worker.isRunning():
#             QMessageBox.warning(self, "警告", "分割正在运行中，无法加载结果图像。")
#             return
#
#         original_image_path, _ = QFileDialog.getOpenFileName(
#             self, "选择原始图像 (用于加载预测结果)", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
#         )
#
#         if not original_image_path:
#             return  # User cancelled
#
#         self.status_label.setText(f"正在加载 {os.path.basename(original_image_path)} 及其预测结果...")
#
#         # Clear previous displays
#         self._clear_all_labels("加载中...")
#
#         try:
#             # Load config to get dataset name for output path construction
#             cfg = {}
#             if os.path.exists(self.current_cfg_path):
#                 with open(self.current_cfg_path) as f:
#                     cfg = yaml.load(f, Loader=yaml.SafeLoader)
#             dataset_name = cfg.get("DATASET", {}).get("dataset", "DummyPolypDataset")  # Default if config missing
#
#             # Determine video_subfolder and base_name from the selected original image path
#             # This logic needs to match how your SegmentationWorker SAVES the files.
#             # Assuming path is something like 'dummy_data/video_001/frame_0000.jpg'
#             # Or a real path like 'C:/YourDataset/video_A/image001.png'
#
#             parts = original_image_path.split(os.sep)
#             file_name = parts[-1]
#             base_name, _ = os.path.splitext(file_name)
#
#             video_subfolder = "unknown_video"  # Default if not found in path
#             for part in parts[:-1]:  # Iterate through parent directories
#                 if 'video_' in part:  # Assuming your video folders are named like 'video_XXX'
#                     video_subfolder = part
#                     break
#
#             if video_subfolder == "unknown_video" and len(parts) >= 2:
#                 # Fallback: if 'video_XXX' pattern not found, assume the immediate parent is the video folder
#                 # This helps if your video images are like 'DatasetName/VideoName/image.jpg'
#                 video_subfolder = parts[-2]
#
#             # Construct expected paths for predicted results
#             base_output_dir = os.path.join(self.current_output_dir, dataset_name, video_subfolder)
#
#             mask_path = os.path.join(base_output_dir, "not_overlay", f"{base_name}_pred.png")
#             overlay_path = os.path.join(base_output_dir, "overlay", f"{base_name}_overlay.png")
#             low_freq_path = os.path.join(base_output_dir, "low_freq", f"{base_name}_low.png")
#             high_freq_path = os.path.join(base_output_dir, "high_freq", f"{base_name}_high.png")
#
#             display_w, display_h = 300, 300  # Match display size
#
#             # Load and display original image
#             orig_img = cv2.imread(original_image_path)
#             if orig_img is None:
#                 QMessageBox.warning(self, "加载失败", f"无法加载原始图像: {original_image_path}")
#                 self._clear_all_labels("无法加载原始图像或预测结果。")
#                 self.status_label.setText("加载失败。")
#                 return
#             orig_qpix = convert_cv_to_qpixmap(orig_img, target_size=(display_w, display_h))
#             self._set_pixmap_or_clear(self.orig_label, orig_qpix, "原始图像")
#
#             # Load and display mask
#             mask_img = cv2.imread(mask_path)
#             if mask_img is None:
#                 print(f"Warning: Mask file not found or could not be loaded: {mask_path}")
#                 self._set_pixmap_or_clear(self.mask_label, QPixmap(), "预测掩码 (未找到)")
#             else:
#                 mask_qpix = convert_cv_to_qpixmap(mask_img, target_size=(display_w, display_h))
#                 self._set_pixmap_or_clear(self.mask_label, mask_qpix, "预测掩码")
#
#             # Load and display overlay
#             if self.checkbox_overlay.isChecked():
#                 overlay_img = cv2.imread(overlay_path)
#                 if overlay_img is None:
#                     print(f"Warning: Overlay file not found or could not be loaded: {overlay_path}")
#                     self._set_pixmap_or_clear(self.overlay_label, QPixmap(), "叠加图像 (未找到)")
#                 else:
#                     overlay_qpix = convert_cv_to_qpixmap(overlay_img, target_size=(display_w, display_h))
#                     self._set_pixmap_or_clear(self.overlay_label, overlay_qpix, "叠加图像")
#             else:
#                 self.overlay_label.clear()
#                 self.overlay_label.setText("已禁用叠加图像显示")
#
#             # Load and display frequency maps
#             if not self.checkbox_no_freq_maps.isChecked():
#                 low_freq_img = cv2.imread(low_freq_path, cv2.IMREAD_GRAYSCALE)  # Freq maps are usually grayscale
#                 high_freq_img = cv2.imread(high_freq_path, cv2.IMREAD_GRAYSCALE)
#
#                 if low_freq_img is None:
#                     print(f"Warning: Low frequency map not found or could not be loaded: {low_freq_path}")
#                     self._set_pixmap_or_clear(self.low_freq_label, QPixmap(), "低频图 (未找到)")
#                 else:
#                     low_freq_qpix = convert_cv_to_qpixmap(low_freq_img, target_size=(display_w, display_h))
#                     self._set_pixmap_or_clear(self.low_freq_label, low_freq_qpix, "低频图")
#
#                 if high_freq_img is None:
#                     print(f"Warning: High frequency map not found or could not be loaded: {high_freq_path}")
#                     self._set_pixmap_or_clear(self.high_freq_label, QPixmap(), "高频图 (未找到)")
#                 else:
#                     high_freq_qpix = convert_cv_to_qpixmap(high_freq_img, target_size=(display_w, display_h))
#                     self._set_pixmap_or_clear(self.high_freq_label, high_freq_qpix, "高频图")
#             else:
#                 self.low_freq_label.clear()
#                 self.high_freq_label.clear()
#                 self.low_freq_label.setText("已禁用低频图显示")
#                 self.high_freq_label.setText("已禁用高频图显示")
#
#             self.status_label.setText(f"已加载 {os.path.basename(original_image_path)} 的结果。")
#
#         except Exception as e:
#             error_traceback = traceback.format_exc()
#             QMessageBox.critical(self, "加载错误", f"加载预测结果时发生错误: {e}\n{error_traceback}")
#             self._clear_all_labels(f"加载错误: {e}")
#             self.status_label.setText("加载失败。")
#
#     # MODIFIED: Function to select and display original and a chosen predicted image for comparison
#     @pyqtSlot()
#     def select_original_and_predicted_for_comparison(self):
#         if self.worker and self.worker.isRunning():
#             QMessageBox.warning(self, "警告", "分割正在运行中，无法加载图像。")
#             return
#
#         self.status_label.setText("请选择原始图像...")
#         original_image_path, _ = QFileDialog.getOpenFileName(
#             self, "选择原始图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
#         )
#
#         if not original_image_path:
#             self.status_label.setText("操作取消。")
#             self._clear_all_labels("")
#             return  # User cancelled
#
#         self.status_label.setText("请选择要对比的预测图像...")
#         predicted_image_path, _ = QFileDialog.getOpenFileName(
#             self, "选择预测图像 (如掩码或叠加图)", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
#         )
#
#         if not predicted_image_path:
#             self.status_label.setText("操作取消。")
#             self._clear_all_labels("")
#             return  # User cancelled
#
#         self.status_label.setText(
#             f"正在显示原始图像: {os.path.basename(original_image_path)} 和预测图像: {os.path.basename(predicted_image_path)}...")
#
#         # Clear all display labels first
#         self._clear_all_labels("加载中...")
#
#         try:
#             display_w, display_h = 300, 300  # Match display size
#
#             # Load and display original image
#             orig_img = cv2.imread(original_image_path)
#             if orig_img is None:
#                 QMessageBox.warning(self, "加载失败", f"无法加载原始图像: {original_image_path}")
#                 self._clear_all_labels("无法加载原始图像。")
#                 self.status_label.setText("加载失败。")
#                 return
#             orig_qpix = convert_cv_to_qpixmap(orig_img, target_size=(display_w, display_h))
#             self._set_pixmap_or_clear(self.orig_label, orig_qpix, "原始图像")
#
#             # Load and display predicted image (mask/overlay)
#             pred_img = cv2.imread(predicted_image_path)
#             if pred_img is None:
#                 QMessageBox.warning(self, "加载失败", f"无法加载预测图像: {predicted_image_path}")
#                 # Still display original if it loaded
#                 self._set_pixmap_or_clear(self.mask_label, QPixmap(), "无法加载预测图像。")
#                 self.status_label.setText("加载失败。")
#                 return
#             pred_qpix = convert_cv_to_qpixmap(pred_img, target_size=(display_w, display_h))
#             self._set_pixmap_or_clear(self.mask_label, pred_qpix, "预测图像")
#
#             # Clear other labels and set placeholder text
#             self.overlay_label.clear()
#             self.low_freq_label.clear()
#             self.high_freq_label.clear()
#             self.overlay_label.setText("未加载叠加图像 (此功能仅用于原始与预测对比)")
#             self.low_freq_label.setText("未加载低频图 (此功能仅用于原始与预测对比)")
#             self.high_freq_label.setText("未加载高频图 (此功能仅用于原始与预测对比)")
#
#             self.status_label.setText(f"已显示原始图像和预测图像进行对比。")
#
#         except Exception as e:
#             error_traceback = traceback.format_exc()
#             QMessageBox.critical(self, "显示错误", f"显示图像时发生错误: {e}\n{error_traceback}")
#             self._clear_all_labels(f"显示错误: {e}")
#             self.status_label.setText("显示失败。")
#
#     def _clear_all_labels(self, text=""):
#         self.orig_label.clear()
#         self.mask_label.clear()
#         self.overlay_label.clear()
#         self.low_freq_label.clear()
#         self.high_freq_label.clear()
#
#         self.orig_label.setText(text)
#         self.mask_label.setText(text)
#         self.overlay_label.setText(text)
#         self.low_freq_label.setText(text)
#         self.high_freq_label.setText(text)
#
#     def closeEvent(self, event):
#         if self.worker and self.worker.isRunning():
#             reply = QMessageBox.question(self, '确认退出',
#                                          "分割仍在运行中。您确定要退出吗？",
#                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
#             if reply == QMessageBox.Yes:
#                 self.worker.stop()
#                 self.worker.wait(5000)
#                 if self.worker.isRunning():
#                     print("工作线程未终止，强制退出。")
#                 event.accept()
#             else:
#                 event.ignore()
#         else:
#             event.accept()
#
#
# if __name__ == "__main__":
#     # Create dummy directories and a dummy image for initial setup
#     os.makedirs("configs", exist_ok=True)
#     os.makedirs("checkpoints/xxx/xx/New_PolypVideoDataset_15", exist_ok=True)  # Specific path
#
#     # Create dummy video folders and images for the dummy prediction worker
#     # This helps ensure the paths exist for saving and later loading
#     for i in range(1, 6):  # Corresponds to 5 dummy videos in DummyDataset
#         os.makedirs(f"dummy_data/video_{i:03d}", exist_ok=True)
#         # Create a few dummy frames in each dummy video folder
#         for j in range(3):  # Create first 3 frames for each video
#             dummy_img_path = f"dummy_data/video_{i:03d}/frame_{j:04d}.jpg"
#             if not os.path.exists(dummy_img_path):
#                 img_content = np.zeros((256, 256, 3), dtype=np.uint8)
#                 cv2.putText(img_content, f"Video {i} Frame {j}", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                             (255, 255, 255), 2)
#                 cv2.imwrite(dummy_img_path, img_content)
#
#     dummy_cfg_path = "configs/New_PolypVideoDataset.yaml"
#     if not os.path.exists(dummy_cfg_path):
#         with open(dummy_cfg_path, "w") as f:
#             f.write("""
# DATASET:
#   dataset: "DummyPolypDataset"
#   Low_mean: [0.5]
#   Low_std: [0.5]
#   High_mean: [0.5]
#   High_std: [0.5]
#   clip_len_pred: 8
#   stride_pred: 8
#   frames_per_video: 100 # Important for dummy dataset length
# PREDICT:
#   batch_size: 1
# TRAIN:
#   num_workers: 0
# """)
#
#     dummy_ckpt_path = "checkpoints/xxx/xx/New_PolypVideoDataset_15/New_PolypVideoDataset_15.ckpt"
#     if not os.path.exists(dummy_ckpt_path):
#         # Create a dummy checkpoint file (its content doesn't matter for dummy trainer)
#         # For a real model, this would be a torch.save() of a state_dict
#         with open(dummy_ckpt_path, "w") as f:
#             f.write("DUMMY_CHECKPOINT_CONTENT")
#
#     app = QApplication(sys.argv)
#     window = VideoSegmentationApp()
#     window.show()
#     sys.exit(app.exec_())



























































import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QCheckBox, QProgressBar,
    QMessageBox, QTabWidget, QSplitter, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

import traceback
import yaml
import warnings

warnings.filterwarnings("ignore")

# --- Global Flag for PyTorch/Lightning Availability ---
_TORCH_AVAILABLE = False
try:
    import torch
    import pytorch_lightning as pl

    _TORCH_AVAILABLE = True
    print("PyTorch and PyTorch Lightning detected, but running prediction in dummy simulation mode for GUI.")
except ImportError:
    print("PyTorch or PyTorch Lightning not found. Running in full dummy simulation mode.")


    class DummyPytorchLightningModule:
        def predict_step(self, batch, batch_idx):
            pass


    pl = type('pl_module', (), {'LightningModule': DummyPytorchLightningModule})()


    class DummyTorch:
        def from_numpy(self, x): return x

        def save(self, obj, f): pass

        def Tensor(self, *args, **kwargs): return object()

        @property
        def cuda(self):
            class DummyCuda:
                def is_available(self): return False

            return DummyCuda()


    torch = DummyTorch()


# --- Mock/Dummy Implementations for the GUI to function without real models ---

class DummyTrainer:
    def __init__(self, devices, accelerator, logger=False, enable_model_summary=False, enable_progress_bar=False):
        print(f"DummyTrainer: Initialized (always simulating, ignoring real device/accelerator settings).")

    def predict(self, model, dataloaders, ckpt_path=None):
        print(f"DummyTrainer: Simulating prediction. (Checkpoint path: '{ckpt_path}' will NOT be loaded).")
        all_predictions = []
        for batch_idx, batch in enumerate(dataloaders):
            prediction_output = model.predict_step(batch, batch_idx)
            all_predictions.append(prediction_output)
        return all_predictions


class DummyDataset:
    CLASSES = ["background", "polyp", "instrument"]
    frames_per_video = 100

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return idx


def create_dummy_data_loader(cfg, dataset_instance):
    clip_len_pred = cfg["DATASET"].get("clip_len_pred", 8)
    stride_pred = cfg["DATASET"].get("stride_pred", clip_len_pred)
    batch_size = cfg["PREDICT"].get("batch_size", 1)

    if not _TORCH_AVAILABLE:
        class MockDataLoader:
            def __init__(self, dataset, sampler, batch_size, num_workers, pin_memory):
                self._len = len(sampler)
                print(f"MockDataLoader: Initialized with {self._len} batches.")

            def __len__(self): return self._len

            def __iter__(self):
                for i in range(self._len):
                    yield i

        DataLoader = MockDataLoader
    else:
        from torch.utils.data import DataLoader

    sampler = SlidingWindowClipSampler(
        dataset=dataset_instance,
        clip_len=clip_len_pred,
        stride=stride_pred,
        shuffle=False,
        drop_last=False
    )
    num_clips = len(sampler)
    print(
        f"SlidingWindowClipSampler: Initialized with {dataset_instance.frames_per_video * len(dataset_instance)} total frames, {num_clips} conceptual clips.")

    dummy_loader = DataLoader(dataset_instance, sampler=sampler, batch_size=batch_size, num_workers=0, pin_memory=False)
    return dummy_loader


def build_model(args, cfg):
    print("Building dummy model (always generating numpy arrays)...")

    class DummyModel(pl.LightningModule):
        def __init__(self, cfg_for_model, class_list):
            super().__init__()
            self.cfg = cfg_for_model
            self.class_list = class_list

        def predict_step(self, batch, batch_idx):
            frames_in_clip = self.cfg["DATASET"].get("clip_len_pred", 8)

            dummy_preds = np.random.randint(0, len(self.class_list), (frames_in_clip, 256, 256), dtype=np.uint8)

            dummy_img_paths = []
            dummy_video_dir = os.path.join("dummy_data",
                                           f"video_{batch_idx + 1:03d}")  # Ensure unique video dirs for saving
            os.makedirs(dummy_video_dir, exist_ok=True)

            for i in range(frames_in_clip):
                frame_idx_in_sequence = batch_idx * frames_in_clip + i
                path = os.path.join(dummy_video_dir, f"frame_{frame_idx_in_sequence:04d}.jpg")
                if not os.path.exists(path):
                    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
                    color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
                    cv2.rectangle(dummy_img, (50, 50), (200, 200), color, -1)
                    cv2.putText(dummy_img, f"Clip {batch_idx} Frame {i}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imwrite(path, dummy_img)
                dummy_img_paths.append(path)

            dummy_low_freqs = np.random.rand(frames_in_clip, 1, 128, 128).astype(np.float32)
            dummy_high_freqs = np.random.rand(frames_in_clip, 1, 128, 128).astype(np.float32)

            return {
                "preds": dummy_preds,
                "img_paths": dummy_img_paths,
                "low_freqs": dummy_low_freqs,
                "high_freqs": dummy_high_freqs
            }

    return DummyModel(cfg, args.class_list)


class SlidingWindowClipSampler:
    def __init__(self, dataset, clip_len, stride, shuffle, drop_last):
        self.dataset = dataset
        self.clip_len = clip_len
        self.stride = stride
        self.shuffle = shuffle
        self.drop_last = drop_last

        total_frames = getattr(dataset, 'frames_per_video', 100) * len(dataset)

        if total_frames < clip_len:
            self._length = 0
        else:
            self._length = (total_frames - clip_len) // stride + 1
            if (total_frames - clip_len) % stride != 0:
                self._length += 1
            if self._length < 0:
                self._length = 0

    def __iter__(self):
        for i in range(len(self)):
            yield i

    def __len__(self):
        return self._length


# --- End Mock/Dummy Implementations ---


def convert_cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None or cv_img.size == 0:
        return QPixmap()

    h, w = cv_img.shape[:2]

    if len(cv_img.shape) == 2:
        bytes_per_line = w
        if cv_img.dtype != np.uint8:
            cv_img = np.clip(cv_img, 0, 255).astype(np.uint8)
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    else:
        ch = cv_img.shape[2]
        if ch == 1:
            if cv_img.dtype != np.uint8:
                cv_img = np.clip(cv_img, 0, 255).astype(np.uint8)
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            bytes_per_line = ch * w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)

    if target_size:
        return QPixmap.fromImage(q_img).scaled(target_size[0], target_size[1], Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation)
    return QPixmap.fromImage(q_img)


class SegmentationWorker(QThread):
    frame_processed = pyqtSignal(QPixmap, QPixmap, QPixmap, QPixmap, QPixmap, int, int)
    segmentation_finished = pyqtSignal()
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, config_path, checkpoint_path, output_pred_path,
                 overlay_enabled, no_freq_maps_enabled, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_pred_path = output_pred_path
        self.overlay_enabled = overlay_enabled
        self.no_freq_maps_enabled = no_freq_maps_enabled
        self._is_running = True

        self.cfg = None
        self.model = None
        self.val_set = None
        self.data_loader_val = None
        self.color_map = {
            0: (0, 0, 0),  # background Black (RGB)
            1: (0, 255, 0),  # Polyp Green (RGB)
            2: (0, 0, 255),  # instrument Blue (RGB)
        }

    def stop(self):
        self._is_running = False

    def run(self):
        self.status_update.emit("正在初始化分割工作器...")
        try:
            # 1. Load Configuration
            if not os.path.exists(self.config_path):
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                dummy_cfg_content = """
DATASET:
  dataset: "DummyPolypDataset"
  Low_mean: [0.5]
  Low_std: [0.5]
  High_mean: [0.5]
  High_std: [0.5]
  clip_len_pred: 8
  stride_pred: 8
  frames_per_video: 100
PREDICT:
  batch_size: 1
TRAIN:
  num_workers: 0
                """
                with open(self.config_path, "w") as f:
                    f.write(dummy_cfg_content)
                self.status_update.emit(f"警告: 配置文件未找到，已创建虚拟配置文件: {self.config_path}")

            with open(self.config_path) as f:
                self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
            self.status_update.emit("配置加载成功。")

            if "PREDICT" not in self.cfg:
                self.cfg["PREDICT"] = {"batch_size": 1}
                self.status_update.emit("警告: 配置文件缺少 'PREDICT' 部分，已添加默认值。")

            # 2. Build Dataset and DataLoader (Always use dummy implementations)
            class DummyArgs:
                def __init__(self, cfg_data, checkpoint_path_val, output_pred_path_val, overlay_enabled_val,
                             no_freq_maps_enabled_val):
                    self.cfg = cfg_data
                    self.checkpoint = checkpoint_path_val
                    self.output_pred = output_pred_path_val
                    self.overlay = overlay_enabled_val
                    self.no_freq_maps = no_freq_maps_enabled_val
                    self.class_list = None

            temp_args = DummyArgs(self.cfg, self.checkpoint_path, self.output_pred_path, self.overlay_enabled,
                                  self.no_freq_maps_enabled)

            self.status_update.emit("正在构建验证数据集 (模拟)...")
            self.val_set = DummyDataset()
            if not self.val_set or len(self.val_set) == 0:
                raise ValueError("验证数据集为空或构建失败。")
            temp_args.class_list = self.val_set.CLASSES
            self.status_update.emit(f"验证数据集构建完成。模拟视频/分段数量: {len(self.val_set)}")

            self.data_loader_val = create_dummy_data_loader(self.cfg, self.val_set)

            if len(self.data_loader_val) == 0:
                raise ValueError("数据加载器为空。无法进行预测。")
            self.status_update.emit(f"数据加载器已创建。批次（剪辑）数量: {len(self.data_loader_val)}")

            # 3. Build Model (Always use DummyModel)
            self.status_update.emit("正在构建模型 (模拟)...")
            self.model = build_model(temp_args, self.cfg)
            self.status_update.emit("模型构建成功。")

            # 4. Setup Trainer (Always use DummyTrainer)
            self.status_update.emit("正在设置模拟 Trainer...")
            trainer = DummyTrainer(
                devices=1,
                accelerator="cpu",
                logger=False,
                enable_model_summary=False,
                enable_progress_bar=False
            )
            self.status_update.emit("模拟 Trainer 已就绪。")

            # 5. Perform Prediction (using DummyTrainer, will not load real checkpoint)
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                with open(self.checkpoint_path, "w") as f:
                    f.write("DUMMY_CHECKPOINT_CONTENT_IGNORED_BY_DUMMY_TRAINER")
                self.status_update.emit(
                    f"警告: 检查点文件未找到，已创建虚拟检查点占位符: {self.checkpoint_path}。此文件内容不影响模拟。")

            self.status_update.emit(f"\n--- 正在进行预测 (模拟模式) ---")
            predictions = trainer.predict(self.model, dataloaders=self.data_loader_val, ckpt_path=self.checkpoint_path)

            if predictions is None:
                raise RuntimeError("模拟预测结果为 None。")
            elif not isinstance(predictions, list):
                raise TypeError(f"模拟预测返回类型 {type(predictions)}, 预期为列表。")
            elif len(predictions) == 0:
                self.error_occurred.emit("警告: 模拟预测返回空列表。没有生成预测结果。")
                self.segmentation_finished.emit()
                return
            else:
                self.status_update.emit(f"\n模拟预测完成。获得 {len(predictions)} 个预测项（剪辑）。")

            # 6. Process and Save Predictions
            self.status_update.emit("--- 正在处理并显示结果 ---")

            total_frames_to_process = sum([item["preds"].shape[0] for item in predictions if "preds" in item])
            if total_frames_to_process == 0:
                self.status_update.emit("没有要处理的帧。")
                self.segmentation_finished.emit()
                return

            processed_frames_count = 0
            # Ensure the base output directory for this dataset is created
            dataset_name = self.cfg["DATASET"].get("dataset", "unknown_dataset")
            save_base_path = os.path.join(self.output_pred_path, dataset_name)
            os.makedirs(save_base_path, exist_ok=True)

            for item_idx, item in enumerate(predictions):
                if not self._is_running:
                    self.status_update.emit("用户已停止分割。")
                    break

                if not isinstance(item, dict) or not all(
                        k in item for k in ["preds", "img_paths", "low_freqs", "high_freqs"]):
                    self.error_occurred.emit(f"警告 [剪辑 {item_idx}]: 意外的项目格式。跳过。")
                    skipped_frames_in_item = item.get("preds", np.empty((0, 0, 0))).shape[0]
                    if not skipped_frames_in_item:
                        skipped_frames_in_item = self.cfg["DATASET"].get("clip_len_pred", 8)
                    processed_frames_count += skipped_frames_in_item
                    percentage = int((processed_frames_count / total_frames_to_process) * 100)
                    if percentage > 100: percentage = 100
                    self.progress_update.emit(percentage)
                    continue

                try:
                    num_frames = item["preds"].shape[0]
                    all_preds = item["preds"].astype(np.uint8)
                    all_img_paths = item["img_paths"]
                    all_low_freqs = item["low_freqs"]
                    all_high_freqs = item["high_freqs"]
                except Exception as e:
                    self.error_occurred.emit(f"错误: 从项目 {item_idx} 解包数据失败: {e}。跳过剪辑。")
                    skipped_frames_in_item = self.cfg["DATASET"].get("clip_len_pred", 8)
                    processed_frames_count += skipped_frames_in_item
                    percentage = int((processed_frames_count / total_frames_to_process) * 100)
                    if percentage > 100: percentage = 100
                    self.progress_update.emit(percentage)
                    continue

                if not (len(all_img_paths) == num_frames and all_low_freqs.shape[0] == num_frames and
                        all_high_freqs.shape[0] == num_frames):
                    self.error_occurred.emit(f"警告 [剪辑 {item_idx}]: 数据长度不匹配。跳过剪辑。")
                    processed_frames_count += num_frames
                    percentage = int((processed_frames_count / total_frames_to_process) * 100)
                    if percentage > 100: percentage = 100
                    self.progress_update.emit(percentage)
                    continue

                self.status_update.emit(f"正在处理剪辑 {item_idx + 1}/{len(predictions)}, 共 {num_frames} 帧...")

                for i in range(num_frames):
                    if not self._is_running:
                        self.status_update.emit("用户已停止分割。")
                        break

                    pred = all_preds[i]
                    img_path = all_img_paths[i]

                    if isinstance(img_path, list):
                        if len(img_path) == 1 and isinstance(img_path[0], str):
                            img_path = img_path[0]
                        else:
                            self.error_occurred.emit(
                                f"警告 [剪辑 {item_idx}, 帧 {i}]: img_path 是列表但不是单个字符串: {img_path}。跳过。")
                            processed_frames_count += 1
                            percentage = int((processed_frames_count / total_frames_to_process) * 100)
                            if percentage > 100: percentage = 100
                            self.progress_update.emit(percentage)
                            continue
                    elif not isinstance(img_path, str):
                        self.error_occurred.emit(
                            f"警告 [剪辑 {item_idx}, 帧 {i}]: img_path 不是字符串: {img_path}。跳过。")
                        processed_frames_count += 1
                        percentage = int((processed_frames_count / total_frames_to_process) * 100)
                        if percentage > 100: percentage = 100
                        self.progress_update.emit(percentage)
                        continue

                    img_path = os.path.normpath(img_path)

                    orig_img = cv2.imread(img_path)
                    if orig_img is None:
                        self.error_occurred.emit(
                            f"警告 [剪辑 {item_idx}, 帧 {i}]: 无法加载图像: {img_path}。请检查文件是否存在和损坏。跳过。")
                        processed_frames_count += 1
                        percentage = int((processed_frames_count / total_frames_to_process) * 100)
                        if percentage > 100: percentage = 100
                        self.progress_update.emit(percentage)
                        continue

                    pred_h, pred_w = pred.shape

                    color_mask_bgr = np.zeros((pred_h, pred_w, 3), dtype=np.uint8)
                    for class_id, color_rgb in self.color_map.items():
                        color_bgr = tuple(reversed(color_rgb))
                        color_mask_bgr[pred == class_id] = color_bgr

                    display_w, display_h = 300, 300

                    # Original and Mask are always scaled to 300x300 for consistency in prediction display
                    orig_qpix = convert_cv_to_qpixmap(orig_img, target_size=(display_w, display_h))
                    mask_qpix = convert_cv_to_qpixmap(color_mask_bgr, target_size=(display_w, display_h))

                    overlay_qpix = QPixmap()
                    if self.overlay_enabled:
                        orig_img_resized = cv2.resize(orig_img, (pred_w, pred_h))
                        overlay_img = cv2.addWeighted(orig_img_resized, 0.6, color_mask_bgr, 0.4, 0)
                        overlay_qpix = convert_cv_to_qpixmap(overlay_img, target_size=(display_w, display_h))

                    low_freq_qpix = QPixmap()
                    high_freq_qpix = QPixmap()
                    if not self.no_freq_maps_enabled:
                        mean_low = float(self.cfg["DATASET"]["Low_mean"][0]) if isinstance(
                            self.cfg["DATASET"]["Low_mean"], (list, tuple)) else float(self.cfg["DATASET"]["Low_mean"])
                        std_low = float(self.cfg["DATASET"]["Low_std"][0]) if isinstance(self.cfg["DATASET"]["Low_std"],
                                                                                         (list, tuple)) else float(
                            self.cfg["DATASET"]["Low_std"])

                        low_freq_data = all_low_freqs[i].squeeze(0) if all_low_freqs[i].ndim > 2 else all_low_freqs[i]
                        denorm_low_freq = (low_freq_data * std_low + mean_low) * 255.0
                        low_freq_resized = cv2.resize(denorm_low_freq, (pred_w, pred_h))
                        low_freq_resized = np.clip(low_freq_resized, 0, 255).astype(np.uint8)
                        low_freq_qpix = convert_cv_to_qpixmap(low_freq_resized, target_size=(display_w, display_h))

                        mean_high = float(self.cfg["DATASET"]["High_mean"][0]) if isinstance(
                            self.cfg["DATASET"]["High_mean"], (list, tuple)) else float(
                            self.cfg["DATASET"]["High_mean"])
                        std_high = float(self.cfg["DATASET"]["High_std"][0]) if isinstance(
                            self.cfg["DATASET"]["High_std"], (list, tuple)) else float(self.cfg["DATASET"]["High_std"])

                        high_freq_data = all_high_freqs[i].squeeze(0) if all_high_freqs[i].ndim > 2 else all_high_freqs[
                            i]
                        denorm_high_freq = (high_freq_data * std_high + mean_high) * 255.0
                        high_freq_resized = cv2.resize(denorm_high_freq, (pred_w, pred_h))
                        high_freq_resized = np.clip(high_freq_resized, 0, 255).astype(np.uint8)
                        high_freq_qpix = convert_cv_to_qpixmap(high_freq_resized, target_size=(display_w, display_h))

                    processed_frames_count += 1
                    percentage = int((processed_frames_count / total_frames_to_process) * 100)
                    if percentage > 100: percentage = 100

                    self.frame_processed.emit(
                        orig_qpix, mask_qpix, overlay_qpix, low_freq_qpix, high_freq_qpix,
                        processed_frames_count, total_frames_to_process
                    )
                    self.progress_update.emit(percentage)

                    # Determine video_subfolder from img_path
                    parts = img_path.split(os.sep)
                    video_subfolder = "unknown_video"
                    for part in parts:  # Iterate through parts to find the 'video_XXX' segment
                        if 'video_' in part:
                            video_subfolder = part
                            break
                    if video_subfolder == "unknown_video":
                        # Fallback for paths that don't match 'video_XXX' pattern
                        # Try to use the grandparent directory name if img_path is like /some/dir/image.jpg
                        # This assumes images are in subfolders directly under the dataset's video path
                        # Example: dummy_data/video_001/frame_0000.jpg -> video_001
                        # If the original path is simply 'frame_0000.jpg', it might default to 'unknown_video'
                        if len(parts) >= 2:
                            video_subfolder = parts[-2] if os.path.isdir(
                                os.path.join(os.path.dirname(img_path), parts[-2])) else "unknown_video"

                    file_name = os.path.basename(img_path)
                    base_name, _ = os.path.splitext(file_name)

                    current_output_sub_dir = os.path.join(save_base_path, video_subfolder)
                    os.makedirs(current_output_sub_dir, exist_ok=True)

                    mask_output_dir = os.path.join(current_output_sub_dir, "not_overlay")
                    os.makedirs(mask_output_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(mask_output_dir, f"{base_name}_pred.png"), color_mask_bgr)

                    if self.overlay_enabled:
                        orig_img_resized = cv2.resize(orig_img, (pred_w, pred_h))
                        overlay_img_to_save = cv2.addWeighted(orig_img_resized, 0.6, color_mask_bgr, 0.4, 0)
                        overlay_output_dir = os.path.join(current_output_sub_dir, "overlay")
                        os.makedirs(overlay_output_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(overlay_output_dir, f"{base_name}_overlay.png"), overlay_img_to_save)

                    if not self.no_freq_maps_enabled:
                        low_freq_output_dir = os.path.join(current_output_sub_dir, "low_freq")
                        high_freq_output_dir = os.path.join(current_output_sub_dir, "high_freq")
                        os.makedirs(low_freq_output_dir, exist_ok=True)
                        os.makedirs(high_freq_output_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(low_freq_output_dir, f"{base_name}_low.png"), low_freq_resized)
                        cv2.imwrite(os.path.join(high_freq_output_dir, f"{base_name}_high.png"), high_freq_resized)

        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"ERROR: An unhandled exception occurred in SegmentationWorker:\n{error_traceback}")
            self.error_occurred.emit(f"分割过程中发生错误: {error_traceback}")
        finally:
            self.segmentation_finished.emit()
            self.status_update.emit("分割过程已完成或终止。")


class VideoSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自然腔道内窥镜影像感知软件V1.0")
        self.setGeometry(100, 100, 1200, 800)

        self.worker = None

        self.init_ui()

        # Set the correct default paths as per your request
        self.current_cfg_path = "configs/New_PolypVideoDataset.yaml"
        self.current_checkpoint_path = "checkpoints/xxx/xx/New_PolypVideoDataset_15/New_PolypVideoDataset_15.ckpt"
        self.current_output_dir = "./pred2"  # Default output directory

        self.update_path_labels()

        self.setStyleSheet("""
            QMainWindow {
                background-color: #2e2e2e;
                color: #e0e0e0;
            }
            QWidget#ControlPanel {
                background-color: #383838;
                border-right: 1px solid #444;
            }
            QPushButton {
                background-color: #4CAF50; /* Green */
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QPushButton#StopButton {
                background-color: #f44336; /* Red */
            }
            QPushButton#StopButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #aaa;
            }
            QLineEdit {
                background-color: #3e3e3e;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
            }
            QLabel[text^="<h4>"] {
                color: #87CEEB; /* Light Blue */
                font-size: 16px;
                font-weight: bold;
                padding-top: 10px;
                padding-bottom: 5px;
            }
            QCheckBox {
                color: #e0e0e0;
                font-size: 13px;
                padding: 5px 0;
            }
            QProgressBar {
                text-align: center;
                color: white;
                background-color: #444;
                border-radius: 7px;
                height: 25px;
                margin: 10px 0;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:1 #007bff);
                border-radius: 7px;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2e2e2e;
            }
            QTabBar::tab {
                background: #3e3e3e;
                color: #e0e0e0;
                padding: 10px 15px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #555;
                color: white;
                border-top: 2px solid #007bff;
            }
            QTabBar::tab:hover {
                background: #4a4a4a;
            }
            QLabel#image_display_label {
                border: 1px dashed #666;
                background-color: #3e3e3e;
                font-size: 16px;
                font-weight: bold;
                color: #aaaaaa;
                qproperty-alignment: AlignCenter;
            }
            QStatusBar {
                background-color: #383838;
                color: #e0e0e0;
                font-size: 12px;
                padding: 5px;
            }
            QSplitter::handle {
                background-color: #444;
                width: 5px;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 2ex; /* Space for the title */
                color: #e0e0e0;
                font-size: 14px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left; /* Position at top left */
                padding: 0 3px;
                background-color: #383838; /* Match panel background */
            }
        """)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel = QWidget()
        control_panel.setObjectName("ControlPanel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)

        # Path Configuration Group
        path_group_box = QGroupBox("文件路径配置")
        path_layout = QVBoxLayout(path_group_box)

        path_layout.addWidget(QLabel("配置文件路径:"))
        self.cfg_path_label = QLabel("未选择")
        self.cfg_path_label.setWordWrap(True)
        path_layout.addWidget(self.cfg_path_label)
        self.btn_select_cfg = QPushButton("选择配置文件")
        self.btn_select_cfg.clicked.connect(self.select_config_file)
        path_layout.addWidget(self.btn_select_cfg)

        path_layout.addWidget(QLabel("模型检查点路径:"))
        self.checkpoint_path_label = QLabel("未选择")
        self.checkpoint_path_label.setWordWrap(True)
        path_layout.addWidget(self.checkpoint_path_label)
        self.btn_select_checkpoint = QPushButton("选择模型检查点")
        self.btn_select_checkpoint.clicked.connect(self.select_checkpoint_file)
        path_layout.addWidget(self.btn_select_checkpoint)

        path_layout.addWidget(QLabel("预测结果输出目录:"))
        self.output_dir_label = QLabel("未选择")
        self.output_dir_label.setWordWrap(True)
        path_layout.addWidget(self.output_dir_label)
        self.btn_select_output = QPushButton("选择输出目录")
        self.btn_select_output.clicked.connect(self.select_output_directory)
        path_layout.addWidget(self.btn_select_output)

        control_layout.addWidget(path_group_box)

        # Segmentation Options Group
        options_group_box = QGroupBox("分割选项")
        options_layout = QVBoxLayout(options_group_box)

        self.checkbox_overlay = QCheckBox("显示叠加图像")
        self.checkbox_overlay.setChecked(True)
        options_layout.addWidget(self.checkbox_overlay)

        self.checkbox_no_freq_maps = QCheckBox("禁用频域图显示")
        self.checkbox_no_freq_maps.setChecked(False)
        options_layout.addWidget(self.checkbox_no_freq_maps)

        control_layout.addWidget(options_group_box)

        # Actions Group
        actions_group_box = QGroupBox("控制")
        actions_layout = QVBoxLayout(actions_group_box)

        self.btn_start = QPushButton("开始分割 (运行模型)")
        self.btn_start.clicked.connect(self.start_segmentation)
        actions_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止分割")
        self.btn_stop.setObjectName("StopButton")
        self.btn_stop.clicked.connect(self.stop_segmentation)
        self.btn_stop.setEnabled(False)
        actions_layout.addWidget(self.btn_stop)

        # New button for loading pre-predicted results
        self.btn_load_result = QPushButton("预测结果 ")
        self.btn_load_result.clicked.connect(self.load_and_display_result_image)
        actions_layout.addWidget(self.btn_load_result)

        # MODIFIED: Button for selecting original and predicted image for comparison
        self.btn_select_original_and_predicted = QPushButton("原始图像")
        self.btn_select_original_and_predicted.clicked.connect(self.select_original_and_predicted_for_comparison)
        actions_layout.addWidget(self.btn_select_original_and_predicted)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        actions_layout.addWidget(self.progress_bar)

        control_layout.addWidget(actions_group_box)

        control_layout.addStretch()  # Push everything to the top

        image_display_widget = QWidget()
        image_layout = QVBoxLayout(image_display_widget)
        image_layout.setAlignment(Qt.AlignTop)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(image_display_widget)
        splitter.setSizes([250, 950])
        main_layout.addWidget(splitter)

        self.image_tabs = QTabWidget()
        image_layout.addWidget(self.image_tabs)

        tab1 = QWidget()
        tab1_layout = QHBoxLayout(tab1)
        self.orig_label = QLabel("原始图像")
        self.orig_label.setAlignment(Qt.AlignCenter)
        # self.orig_label.setScaledContents(True) # REMOVE THIS LINE
        self.orig_label.setObjectName("image_display_label")
        self.mask_label = QLabel("预测掩码")
        self.mask_label.setAlignment(Qt.AlignCenter)
        # self.mask_label.setScaledContents(True) # REMOVE THIS LINE
        self.mask_label.setObjectName("image_display_label")
        tab1_layout.addWidget(self.orig_label)
        tab1_layout.addWidget(self.mask_label)
        self.image_tabs.addTab(tab1, "原始 & 掩码")

        tab2 = QWidget()
        tab2_layout = QHBoxLayout(tab2)
        self.overlay_label = QLabel("叠加图像")
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setScaledContents(True) # Keep scaled for overlays/freq maps as they are derived
        self.overlay_label.setObjectName("image_display_label")
        self.low_freq_label = QLabel("低频图")
        self.low_freq_label.setAlignment(Qt.AlignCenter)
        self.low_freq_label.setScaledContents(True) # Keep scaled
        self.low_freq_label.setObjectName("image_display_label")
        self.high_freq_label = QLabel("高频图")
        self.high_freq_label.setAlignment(Qt.AlignCenter)
        self.high_freq_label.setScaledContents(True) # Keep scaled
        self.high_freq_label.setObjectName("image_display_label")

        tab2_layout.addWidget(self.overlay_label)
        tab2_layout.addWidget(self.low_freq_label)
        tab2_layout.addWidget(self.high_freq_label)
        self.image_tabs.addTab(tab2, "叠加 & 频域图")

        self.statusBar = self.statusBar()
        self.status_label = QLabel("准备就绪")
        self.statusBar.addWidget(self.status_label)

    def update_path_labels(self):
        self.cfg_path_label.setText(self.current_cfg_path)
        self.checkpoint_path_label.setText(self.current_checkpoint_path)
        self.output_dir_label.setText(self.current_output_dir)

    @pyqtSlot()
    def select_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "YAML 文件 (*.yaml);;所有文件 (*)")
        if file_path:
            self.current_cfg_path = file_path
            self.cfg_path_label.setText(file_path)

    @pyqtSlot()
    def select_checkpoint_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型检查点", "", "检查点文件 (*.ckpt);;所有文件 (*)")
        if file_path:
            self.current_checkpoint_path = file_path
            self.checkpoint_path_label.setText(file_path)

    @pyqtSlot()
    def select_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录", "./")
        if dir_path:
            self.current_output_dir = dir_path
            self.output_dir_label.setText(dir_path)

    @pyqtSlot()
    def start_segmentation(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "警告", "分割正在运行中。")
            return

        # Try to load config for dataset name
        cfg = {}
        if os.path.exists(self.current_cfg_path):
            try:
                with open(self.current_cfg_path) as f:
                    cfg = yaml.load(f, Loader=yaml.SafeLoader)
            except Exception as e:
                QMessageBox.critical(self, "配置错误", f"无法加载配置文件: {e}")
                return
        else:
            # If config doesn't exist, create a dummy one
            os.makedirs(os.path.dirname(self.current_cfg_path), exist_ok=True)
            dummy_cfg_content = """
DATASET:
  dataset: "DummyPolypDataset"
  Low_mean: [0.5]
  Low_std: [0.5]
  High_mean: [0.5]
  High_std: [0.5]
  clip_len_pred: 8
  stride_pred: 8
  frames_per_video: 100
PREDICT:
  batch_size: 1
TRAIN:
  num_workers: 0
            """
            try:
                with open(self.current_cfg_path, "w") as f:
                    f.write(dummy_cfg_content)
                QMessageBox.information(self, "配置创建",
                                        f"配置文件不存在，已为您创建虚拟配置文件: {self.current_cfg_path}")
                self.cfg_path_label.setText(self.current_cfg_path)
                cfg = yaml.load(dummy_cfg_content, Loader=yaml.SafeLoader)  # Load newly created config
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法创建虚拟配置文件: {e}")
                return

        if not self.current_output_dir:
            QMessageBox.warning(self, "输出目录错误", "请选择一个有效的输出目录。")
            return

        self.orig_label.clear()
        self.mask_label.clear()
        self.overlay_label.clear()
        self.low_freq_label.clear()
        self.high_freq_label.clear()

        self.orig_label.setText("正在加载原始图像...")
        self.mask_label.setText("正在加载预测掩码...")
        self.overlay_label.setText("正在加载叠加图像...")
        self.low_freq_label.setText("正在加载低频图...")
        self.high_freq_label.setText("正在加载高频图...")

        self.progress_bar.setValue(0)
        self.status_label.setText("正在开始分割...")

        self.btn_start.setEnabled(False)
        self.btn_load_result.setEnabled(False)  # Disable load button during segmentation
        self.btn_select_original_and_predicted.setEnabled(False)  # Disable new button
        self.btn_stop.setEnabled(True)

        self.worker = SegmentationWorker(
            self.current_cfg_path,
            self.current_checkpoint_path,
            self.current_output_dir,
            self.checkbox_overlay.isChecked(),
            self.checkbox_no_freq_maps.isChecked()
        )
        self.worker.frame_processed.connect(self.update_display)
        self.worker.segmentation_finished.connect(self.segmentation_finished)
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.error_occurred.connect(self.show_error_message)

        self.worker.start()

    @pyqtSlot()
    def stop_segmentation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_stop.setEnabled(False)
            self.status_label.setText("正在停止分割，请稍候...")
        else:
            QMessageBox.information(self, "信息", "没有正在运行的分割任务。")

    @pyqtSlot(QPixmap, QPixmap, QPixmap, QPixmap, QPixmap, int, int)
    def update_display(self, orig_qpix, mask_qpix, overlay_qpix, low_freq_qpix, high_freq_qpix, current_frame,
                       total_frames):
        # For original and mask labels, we want to respect their actual content size
        # but the pixmap itself is already scaled to 300x300 by convert_cv_to_qpixmap
        # so simply setting it will work with setScaledContents(False) making it centered.
        self._set_pixmap_or_clear(self.orig_label, orig_qpix, "原始图像")
        self._set_pixmap_or_clear(self.mask_label, mask_qpix, "预测掩码")

        # For overlay and frequency maps, keep setScaledContents(True) as they are typically derived
        # and should fill their designated display areas.
        if self.checkbox_overlay.isChecked():
            self._set_pixmap_or_clear(self.overlay_label, overlay_qpix, "叠加图像")
        else:
            self.overlay_label.clear()
            self.overlay_label.setText("已禁用叠加图像显示")

        if not self.checkbox_no_freq_maps.isChecked():
            self._set_pixmap_or_clear(self.low_freq_label, low_freq_qpix, "低频图")
            self._set_pixmap_or_clear(self.high_freq_label, high_freq_qpix, "高频图")
        else:
            self.low_freq_label.clear()
            self.high_freq_label.clear()
            self.low_freq_label.setText("已禁用低频图显示")
            self.high_freq_label.setText("已禁用高频图显示")

        self.status_label.setText(f"正在处理帧 {current_frame}/{total_frames}")

    def _set_pixmap_or_clear(self, label, pixmap, placeholder_text=""):
        if not pixmap.isNull():
            # If setScaledContents(False), pixmap will be centered.
            # If setScaledContents(True), pixmap will scale to fill label.
            # We determine which behavior based on the label's `setScaledContents` property
            # For orig_label and mask_label, it's False. For others, it's True.
            label.setPixmap(pixmap)
            label.setText("")
        else:
            label.clear()
            label.setText(f"无法显示 {placeholder_text}\n(可能无数据或加载失败)")

    @pyqtSlot()
    def segmentation_finished(self):
        self.status_label.setText("分割过程已完成。")
        self.btn_start.setEnabled(True)
        self.btn_load_result.setEnabled(True)
        self.btn_select_original_and_predicted.setEnabled(True)  # Enable new button
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        if self.worker:
            self.worker.quit()
            self.worker.wait()

    @pyqtSlot(str)
    def show_error_message(self, message):
        QMessageBox.critical(self, "错误", message)
        self.status_label.setText("发生错误。")
        self.btn_start.setEnabled(True)
        self.btn_load_result.setEnabled(True)
        self.btn_select_original_and_predicted.setEnabled(True)  # Enable new button
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(0)

        self.orig_label.clear()
        self.mask_label.clear()
        self.overlay_label.clear()
        self.low_freq_label.clear()
        self.high_freq_label.clear()

        display_error = "发生错误，请查看控制台输出。"
        if "\n" in message:
            display_error = f"错误：{message.splitlines()[0]}..."

        self.orig_label.setText(f"无法显示图像\n{display_error}")
        self.mask_label.setText(f"无法显示图像\n{display_error}")
        self.overlay_label.setText(f"无法显示图像\n{display_error}")
        self.low_freq_label.setText(f"无法显示图像\n{display_error}")
        self.high_freq_label.setText(f"无法显示图像\n{display_error}")

        if self.worker:
            self.worker.quit()
            self.worker.wait()

    @pyqtSlot()
    def load_and_display_result_image(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "警告", "分割正在运行中，无法加载结果图像。")
            return

        original_image_path, _ = QFileDialog.getOpenFileName(
            self, "选择原始图像 (用于加载预测结果)", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
        )

        if not original_image_path:
            return  # User cancelled

        self.status_label.setText(f"正在加载 {os.path.basename(original_image_path)} 及其预测结果...")

        # Clear previous displays
        self._clear_all_labels("加载中...")

        try:
            # Load config to get dataset name for output path construction
            cfg = {}
            if os.path.exists(self.current_cfg_path):
                with open(self.current_cfg_path) as f:
                    cfg = yaml.load(f, Loader=yaml.SafeLoader)
            dataset_name = cfg.get("DATASET", {}).get("dataset", "DummyPolypDataset")  # Default if config missing

            # Determine video_subfolder and base_name from the selected original image path
            # This logic needs to match how your SegmentationWorker SAVES the files.
            # Assuming path is something like 'dummy_data/video_001/frame_0000.jpg'
            # Or a real path like 'C:/YourDataset/video_A/image001.png'

            parts = original_image_path.split(os.sep)
            file_name = parts[-1]
            base_name, _ = os.path.splitext(file_name)

            video_subfolder = "unknown_video"  # Default if not found in path
            for part in parts[:-1]:  # Iterate through parent directories
                if 'video_' in part:  # Assuming your video folders are named like 'video_XXX'
                    video_subfolder = part
                    break

            if video_subfolder == "unknown_video" and len(parts) >= 2:
                # Fallback: if 'video_XXX' pattern not found, assume the immediate parent is the video folder
                # This helps if your video images are like 'DatasetName/VideoName/image.jpg'
                video_subfolder = parts[-2]

            # Construct expected paths for predicted results
            base_output_dir = os.path.join(self.current_output_dir, dataset_name, video_subfolder)

            mask_path = os.path.join(base_output_dir, "not_overlay", f"{base_name}_pred.png")
            overlay_path = os.path.join(base_output_dir, "overlay", f"{base_name}_overlay.png")
            low_freq_path = os.path.join(base_output_dir, "low_freq", f"{base_name}_low.png")
            high_freq_path = os.path.join(base_output_dir, "high_freq", f"{base_name}_high.png")

            display_w, display_h = 300, 300  # Match display size for consistency

            # Load and display original image
            orig_img = cv2.imread(original_image_path)
            if orig_img is None:
                QMessageBox.warning(self, "加载失败", f"无法加载原始图像: {original_image_path}")
                self._clear_all_labels("无法加载原始图像或预测结果。")
                self.status_label.setText("加载失败。")
                return
            # Keep original scaling for the "预测结果" tab
            orig_qpix = convert_cv_to_qpixmap(orig_img, target_size=(display_w, display_h))
            self._set_pixmap_or_clear(self.orig_label, orig_qpix, "原始图像")

            # Load and display mask
            mask_img = cv2.imread(mask_path)
            if mask_img is None:
                print(f"Warning: Mask file not found or could not be loaded: {mask_path}")
                self._set_pixmap_or_clear(self.mask_label, QPixmap(), "预测掩码 (未找到)")
            else:
                mask_qpix = convert_cv_to_qpixmap(mask_img, target_size=(display_w, display_h))
                self._set_pixmap_or_clear(self.mask_label, mask_qpix, "预测掩码")

            # Load and display overlay
            if self.checkbox_overlay.isChecked():
                overlay_img = cv2.imread(overlay_path)
                if overlay_img is None:
                    print(f"Warning: Overlay file not found or could not be loaded: {overlay_path}")
                    self._set_pixmap_or_clear(self.overlay_label, QPixmap(), "叠加图像 (未找到)")
                else:
                    overlay_qpix = convert_cv_to_qpixmap(overlay_img, target_size=(display_w, display_h))
                    self._set_pixmap_or_clear(self.overlay_label, overlay_qpix, "叠加图像")
            else:
                self.overlay_label.clear()
                self.overlay_label.setText("已禁用叠加图像显示")

            # Load and display frequency maps
            if not self.checkbox_no_freq_maps.isChecked():
                low_freq_img = cv2.imread(low_freq_path, cv2.IMREAD_GRAYSCALE)  # Freq maps are usually grayscale
                high_freq_img = cv2.imread(high_freq_path, cv2.IMREAD_GRAYSCALE)

                if low_freq_img is None:
                    print(f"Warning: Low frequency map not found or could not be loaded: {low_freq_path}")
                    self._set_pixmap_or_clear(self.low_freq_label, QPixmap(), "低频图 (未找到)")
                else:
                    low_freq_qpix = convert_cv_to_qpixmap(low_freq_img, target_size=(display_w, display_h))
                    self._set_pixmap_or_clear(self.low_freq_label, low_freq_qpix, "低频图")

                if high_freq_img is None:
                    print(f"Warning: High frequency map not found or could not be loaded: {high_freq_path}")
                    self._set_pixmap_or_clear(self.high_freq_label, QPixmap(), "高频图 (未找到)")
                else:
                    high_freq_qpix = convert_cv_to_qpixmap(high_freq_img, target_size=(display_w, display_h))
                    self._set_pixmap_or_clear(self.high_freq_label, high_freq_qpix, "高频图")
            else:
                self.low_freq_label.clear()
                self.high_freq_label.clear()
                self.low_freq_label.setText("已禁用低频图显示")
                self.high_freq_label.setText("已禁用高频图显示")

            self.status_label.setText(f"已加载 {os.path.basename(original_image_path)} 的结果。")

        except Exception as e:
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "加载错误", f"加载预测结果时发生错误: {e}\n{error_traceback}")
            self._clear_all_labels(f"加载错误: {e}")
            self.status_label.setText("加载失败。")

    # MODIFIED: Function to select and display original and a chosen predicted image for comparison
    @pyqtSlot()
    def select_original_and_predicted_for_comparison(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "警告", "分割正在运行中，无法加载图像。")
            return

        self.status_label.setText("请选择原始图像...")
        original_image_path, _ = QFileDialog.getOpenFileName(
            self, "选择原始图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
        )

        if not original_image_path:
            self.status_label.setText("操作取消。")
            self._clear_all_labels("")
            return  # User cancelled

        self.status_label.setText("请选择要对比的预测图像...")
        predicted_image_path, _ = QFileDialog.getOpenFileName(
            self, "选择预测图像 (如掩码或叠加图)", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
        )

        if not predicted_image_path:
            self.status_label.setText("操作取消。")
            self._clear_all_labels("")
            return  # User cancelled

        self.status_label.setText(
            f"正在显示原始图像: {os.path.basename(original_image_path)} 和预测图像: {os.path.basename(predicted_image_path)}...")

        # Clear all display labels first
        self._clear_all_labels("加载中...")

        try:
            # Load and display original image
            orig_img = cv2.imread(original_image_path)
            if orig_img is None:
                QMessageBox.warning(self, "加载失败", f"无法加载原始图像: {original_image_path}")
                self._clear_all_labels("无法加载原始图像。")
                self.status_label.setText("加载失败。")
                return
            # Do not scale here, just convert to QPixmap.
            # The QLabel itself will handle display if setScaledContents(False)
            orig_qpix = convert_cv_to_qpixmap(orig_img, target_size=None) # Pass None to avoid initial scaling

            # Load and display predicted image (mask/overlay)
            pred_img = cv2.imread(predicted_image_path)
            if pred_img is None:
                QMessageBox.warning(self, "加载失败", f"无法加载预测图像: {predicted_image_path}")
                # Still display original if it loaded
                self._set_pixmap_or_clear(self.mask_label, QPixmap(), "无法加载预测图像。")
                self.status_label.setText("加载失败。")
                return
            # Do not scale here, just convert to QPixmap.
            pred_qpix = convert_cv_to_qpixmap(pred_img, target_size=None) # Pass None to avoid initial scaling

            # Determine the maximum size for display within the label
            # This is a common pattern when setScaledContents(False) is used
            max_label_width = self.orig_label.width()
            max_label_height = self.orig_label.height()

            # Scale original image if it's too large for the label
            if not orig_qpix.isNull() and (orig_qpix.width() > max_label_width or orig_qpix.height() > max_label_height):
                orig_qpix = orig_qpix.scaled(max_label_width, max_label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._set_pixmap_or_clear(self.orig_label, orig_qpix, "原始图像")


            # Scale predicted image if it's too large for the label
            if not pred_qpix.isNull() and (pred_qpix.width() > max_label_width or pred_qpix.height() > max_label_height):
                pred_qpix = pred_qpix.scaled(max_label_width, max_label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._set_pixmap_or_clear(self.mask_label, pred_qpix, "预测图像")


            # Clear other labels and set placeholder text
            self.overlay_label.clear()
            self.low_freq_label.clear()
            self.high_freq_label.clear()
            self.overlay_label.setText("未加载叠加图像 (此功能仅用于原始与预测对比)")
            self.low_freq_label.setText("未加载低频图 (此功能仅用于原始与预测对比)")
            self.high_freq_label.setText("未加载高频图 (此功能仅用于原始与预测对比)")

            self.status_label.setText(f"已显示原始图像和预测图像进行对比。")

        except Exception as e:
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "显示错误", f"显示图像时发生错误: {e}\n{error_traceback}")
            self._clear_all_labels(f"显示错误: {e}")
            self.status_label.setText("显示失败。")

    def _clear_all_labels(self, text=""):
        self.orig_label.clear()
        self.mask_label.clear()
        self.overlay_label.clear()
        self.low_freq_label.clear()
        self.high_freq_label.clear()

        self.orig_label.setText(text)
        self.mask_label.setText(text)
        self.overlay_label.setText(text)
        self.low_freq_label.setText(text)
        self.high_freq_label.setText(text)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, '确认退出',
                                         "分割仍在运行中。您确定要退出吗？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait(5000)
                if self.worker.isRunning():
                    print("工作线程未终止，强制退出。")
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    # Create dummy directories and a dummy image for initial setup
    os.makedirs("configs", exist_ok=True)
    os.makedirs("checkpoints/xxx/xx/New_PolypVideoDataset_15", exist_ok=True)  # Specific path

    # Create dummy video folders and images for the dummy prediction worker
    # This helps ensure the paths exist for saving and later loading
    for i in range(1, 6):  # Corresponds to 5 dummy videos in DummyDataset
        os.makedirs(f"dummy_data/video_{i:03d}", exist_ok=True)
        # Create a few dummy frames in each dummy video folder
        for j in range(3):  # Create first 3 frames for each video
            dummy_img_path = f"dummy_data/video_{i:03d}/frame_{j:04d}.jpg"
            if not os.path.exists(dummy_img_path):
                img_content = np.zeros((256, 256, 3), dtype=np.uint8)
                cv2.putText(img_content, f"Video {i} Frame {j}", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
                cv2.imwrite(dummy_img_path, img_content)

    dummy_cfg_path = "configs/New_PolypVideoDataset.yaml"
    if not os.path.exists(dummy_cfg_path):
        with open(dummy_cfg_path, "w") as f:
            f.write("""
DATASET:
  dataset: "DummyPolypDataset"
  Low_mean: [0.5]
  Low_std: [0.5]
  High_mean: [0.5]
  High_std: [0.5]
  clip_len_pred: 8
  stride_pred: 8
  frames_per_video: 100 # Important for dummy dataset length
PREDICT:
  batch_size: 1
TRAIN:
  num_workers: 0
""")

    dummy_ckpt_path = "checkpoints/xxx/xx/New_PolypVideoDataset_15/New_PolypVideoDataset_15.ckpt"
    if not os.path.exists(dummy_ckpt_path):
        # Create a dummy checkpoint file (its content doesn't matter for dummy trainer)
        # For a real model, this would be a torch.save() of a state_dict
        with open(dummy_ckpt_path, "w") as f:
            f.write("DUMMY_CHECKPOINT_CONTENT")

    app = QApplication(sys.argv)
    window = VideoSegmentationApp()
    window.show()
    sys.exit(app.exec_())











