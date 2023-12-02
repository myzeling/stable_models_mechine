import pandas as pd

class PerformanceMetrics:
    
    @staticmethod
    def calculate_Roos(df, args):
        """
        Calculates the out-of-sample R squared (Roos) based on provided dataframe and saves the result to a txt file.
        
        Parameters:
            df (pd.DataFrame): A DataFrame containing the actual and predicted returns.
                               Assumes 'ret' column for actual returns and 'pred' for predictions.
            model_name (str): Name of the model used for prediction.
                               
        Returns:
            float: The Roos value.
        """
        # Calculate the numerator
        if args.formers:
            numerator = ((df['ret_c2c'] - df['pred']) ** 2).sum()
            denominator = (df['ret_c2c'] ** 2).sum()
            Roos = 1 - (numerator / denominator)
        else:
            numerator = ((df['ret_next'] - df['pred']) ** 2).sum()

            # Calculate the denominator
            denominator = (df['ret_next'] ** 2).sum()

            # Calculate Roos
            Roos = 1 - (numerator / denominator)
        
        # Save the Roos value to a txt file named after the model
        with open(f"./{args.result_path}/{args.model}.txt", "w") as file:
            file.write(str(Roos))

        return Roos
