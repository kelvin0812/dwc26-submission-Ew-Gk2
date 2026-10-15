"""Run everything in order. Step 3 isn't here because step 4 imports it."""
import step1_data
import step2_features
import step4_train_and_match
import step5_forecast_and_export

if __name__ == "__main__":
    step1_data.main()
    step2_features.main()
    step4_train_and_match.main()
    step5_forecast_and_export.main()
