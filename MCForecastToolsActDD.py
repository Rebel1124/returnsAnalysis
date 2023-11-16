# Import libraries and dependencies
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go

np.random.seed(42)

class MCSimulation:
    """
    A Python class for runnning Monte Carlo simulation on portfolio returns data. 
    
    ...
    
    Attributes
    ----------
    portfolio_data : pandas.DataFrame
        portfolio dataframe
    weights: list(float)
        portfolio investment breakdown
    nSim: int
        number of samples in simulation
    nTrading: int
        number of trading months to simulate
    simulated_return : pandas.DataFrame
        Simulated data from Monte Carlo
    confidence_interval : pandas.Series
        the 95% confidence intervals for simulated final cumulative returns
        
    """
    
    def __init__(self, portfolio_data, weights="", num_simulation=1000, num_trading_months=12, distributionDD=0.06, lower=0.025, upper=0.975):
        """
        Constructs all the necessary attributes for the MCSimulation object.

        Parameters
        ----------
        portfolio_data: pandas.DataFrame
            DataFrame containing monthly returns information
        weights: list(float)
            A list fractions representing percentage of total investment per stock. DEFAULT: Equal distribution
        num_simulation: int
            Number of simulation samples. DEFAULT: 1000 simulation samples
        num_trading_days: int
            Number of trading days to simulate. DEFAULT: 12 months (1 year of business days)
        """
        
        # Check to make sure that all attributes are set
        if not isinstance(portfolio_data, pd.DataFrame):
            raise TypeError("portfolio_data must be a Pandas DataFrame")
            
        # Set weights if empty, otherwise make sure sum of weights equals one.
        if weights == "":
            num_portfolios = len(portfolio_data.columns.unique())
            weights = [1.0/num_portfolios for s in range(0,num_portfolios)]
        else:
            if round(sum(weights),2) < .99:
                raise AttributeError("Sum of portfolio weights must equal one.")
        
        # Set class attributes
        self.portfolio_data = portfolio_data
        self.weights = weights
        self.nSim = num_simulation
        self.nTrading = num_trading_months
        self.simulated_return = ""
        self.distribution = distributionDD
        self.lower = lower
        self.upper = upper
        
    def calc_cumulative_return(self):
        """
        Calculates the cumulative return over time using a Monte Carlo simulation (Brownian motion with drift).

        """
        num_managers = len(self.portfolio_data.columns.unique())
        last_prices = [1] * num_managers

        monthly_returns = self.portfolio_data

        # Initialize empty Dataframe to hold simulated prices
        portfolio_cumulative_returns = pd.DataFrame()
        
        # Run the simulation of projecting returns 'nSim' number of times
        for n in range(self.nSim):
        
            #if n % 10 == 0:
            #    print(f"Running Monte Carlo simulation number {n}.")
        
            # Create a list of lists to contain the simulated values for each return (manager)
            simvals = [[p] for p in last_prices]
    
            # For each portfolio in our data:
            for s in range(len(last_prices)):

                sequence = monthly_returns[monthly_returns.columns.to_list()[s]].values

                # Simulate the returns for each month
                for i in range(self.nTrading):


                    ddMonths = range(12,self.nTrading,12)
                    drawDown = []
                    for t in ddMonths:
                        drawDown.append(t)

                    if i in drawDown:
                        downDraw=(1-self.distribution)
                    else:
                        downDraw=1

        
                    # Calculate the simulated price using the last price within the list
                    simvals[s].append(simvals[s][-1] * ((1 + random.choices(sequence, k=1)[0])*downDraw))


            # Calculate the monthly returns of simulated prices
            sim_df = pd.DataFrame(simvals).T.pct_change()
    
            # Use the `dot` function with the weights to multiply weights with each column's simulated daily returns
            sim_df = sim_df.dot(self.weights)
    

            # Calculate the normalized, cumulative return series
            portfolio_cumulative_returns[n] = (1 + sim_df.fillna(0)).cumprod()
        
        # Set attribute to use in plotting
        self.simulated_return = portfolio_cumulative_returns
        
        # Calculate 95% confidence intervals for final cumulative returns
        #self.confidence_interval = portfolio_cumulative_returns.iloc[-1, :].quantile(q=[0.025, 0.975])
        self.confidence_interval = portfolio_cumulative_returns.iloc[-1, :].quantile(q=[self.lower, self.upper])
        
        return portfolio_cumulative_returns
    
    def plot_simulation(self):
        """
        Visualizes the simulated stock trajectories using calc_cumulative_return method.

        """ 
        
        # Check to make sure that simulation has run previously. 
        if not isinstance(self.simulated_return,pd.DataFrame):
            self.calc_cumulative_return()


        fig = go.Figure()

        for col in (self.simulated_return.columns.tolist()):
            fig.add_trace(
            go.Scatter(
                x=self.simulated_return.index,
                y=self.simulated_return[col],
            ))


        fig.update_layout(
            title={
                'text': "Cumulative Portfolio Returns",
            },
            template="seaborn",
            xaxis=dict(autorange=True,
                    title_text='Months'),
            yaxis=dict(autorange=True,
                    title_text='Cumulative Return'),
            showlegend=False,
            height=275,
            width=600
        )

        return fig




    
    def plot_distribution(self):
        """
        Visualizes the distribution of cumulative returns simulated using calc_cumulative_return method.

        """
        
        # Check to make sure that simulation has run previously. 
        if not isinstance(self.simulated_return,pd.DataFrame):
            self.calc_cumulative_return()


##
#        # Use the `plot` function to create a probability distribution histogram of simulated ending prices
#        # with markings for a 95% confidence interval
#        plot_title = f"Distribution of Final Cumuluative Returns Across All {self.nSim} Simulations"
#        plt = self.simulated_return.iloc[-1, :].plot(kind='hist', bins=10,density=True,title=plot_title)
#        plt.axvline(self.confidence_interval.iloc[0], color='r')
#        plt.axvline(self.confidence_interval.iloc[1], color='r')
#        return plt
##
    

        fig = go.Figure(data=[go.Histogram(x=self.simulated_return.iloc[-1, :], histnorm='probability')])

        fig.update_layout(
            title={
                'text': "Portfolio Returns Distribuion",
            },
            template="seaborn",
            xaxis=dict(autorange=True,
                    title_text='Ending Values'),
            yaxis=dict(autorange=True,
                    title_text='Probability'),
            height=275,
            width=600

        )

        return fig





    def summarize_cumulative_return(self):
        """
        Calculate final summary statistics for Monte Carlo simulated returns.
        
        """
        
        # Check to make sure that simulation has run previously. 
        if not isinstance(self.simulated_return,pd.DataFrame):
            self.calc_cumulative_return()
            
        metrics = self.simulated_return.iloc[-1].describe()
        ci_series = self.confidence_interval
        ci_series.index = ["CI Lower","CI Upper"]
        return metrics, ci_series