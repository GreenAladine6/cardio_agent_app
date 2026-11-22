# ML Evaluation: Short Answers

## SVM classification

"I use an RBF-kernel Support Vector Classifier to distinguish heart-sound classes, such as normal and pathological. Audio is converted into fixed-length acoustic features, and I standardize the features before fitting the SVM. I compare it with other classifiers using a stratified train/validation/test split. I prioritize pathological-class recall, then weighted F1 and specificity, because missing an abnormal recording matters. The held-out test set is used for the final evaluation."

If asked what the kernel does: "The RBF kernel lets the SVM learn nonlinear class boundaries. The margin is controlled by C, and gamma controls how local each training example's influence is."

## Regression

"Regression predicts a continuous value, while my current heart-sound task predicts categories, so it is classification, not regression. If I needed to predict a continuous measure, such as a score, I could evaluate a regressor such as SVR and report MAE, RMSE, and R-squared on held-out data."

Be clear: the current project does not implement a regression model. SVR is a possible extension, not a model I have already trained.

## LLM fine-tuning

"I use supervised, parameter-efficient fine-tuning with QLoRA, rather than updating all the base-model weights. The configured base is Qwen2.5-7B-Instruct. I load it in 4-bit NF4, train LoRA adapters on selected attention and MLP layers, and track the run with MLflow. The default setup uses rank 16, three epochs, and a 20% validation split."

"The training pipeline formats instruction/input/output examples, checks for duplicate prompts to reduce train-validation leakage, saves the adapter, and supports merging and exporting it as a quantized GGUF model for Ollama."

**Be transparent about data:** the checked-in JSONL is synthetic demonstration data, not a clinically validated dataset. Describe the fine-tuning pipeline as implemented/configured; only claim a real training result if you have run it on an authorized, suitable dataset and can show its evaluation metrics.
