import os
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

def create_shap_force_plot(shap_values, instance, feature_names, model_name, date, output_dir):
    plt.figure(figsize=(1.2, 1.2))
    plt.rcParams.update({'font.size': 6})

    feature_order = np.argsort(np.abs(shap_values[0]))
    top_features = feature_order[-10:]

    plt.barh(
        range(len(top_features)),
        shap_values[0][top_features],
        color=['red' if x > 0 else 'blue' for x in shap_values[0][top_features]],
        edgecolor='black',
        linewidth=1.2
    )

    plt.yticks(
        range(len(top_features)),
        [f"{feature_names[i]} = {instance.iloc[0, i]:.2f}" for i in top_features],
        fontsize=6
    )

    plt.xlabel("SHAP value", fontsize=6)
    plt.title(f"{date}", fontsize=6)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"shap_force_plot_{model_name}_{date}.png"), dpi=600)
    plt.close()


def generate_interpretability_outputs(X, rf_model, xgb_model, indices, output_dir):
    explainer_rf = shap.TreeExplainer(rf_model.named_steps["rf"])
    explainer_xgb = shap.TreeExplainer(xgb_model.named_steps["xgb"])

    lime_explainer = LimeTabularExplainer(
        X.values, feature_names=X.columns, mode="regression"
    )

    os.makedirs(output_dir, exist_ok=True)

    for idx in indices:
        instance = X.loc[idx:idx]

        shap_rf = explainer_rf.shap_values(instance)
        shap_xgb = explainer_xgb.shap_values(instance)

        create_shap_force_plot(shap_rf, instance, X.columns, "RF", idx.date(), output_dir)
        create_shap_force_plot(shap_xgb, instance, X.columns, "XGB", idx.date(), output_dir)

        lime_rf = lime_explainer.explain_instance(instance.values[0], rf_model.predict, num_features=10)
        lime_xgb = lime_explainer.explain_instance(instance.values[0], xgb_model.predict, num_features=10)

        lime_rf.save_to_file(os.path.join(output_dir, f"lime_rf_{idx.date()}.html"))
        lime_xgb.save_to_file(os.path.join(output_dir, f"lime_xgb_{idx.date()}.html"))
