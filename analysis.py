import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import streamlit as st
st.set_page_config(layout="wide")
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize as optimize
from PIL import Image
import json
from streamlit_lottie import st_lottie

from MCForecastToolsActDD import MCSimulation

np.random.seed(6)

#file = "returnsData.csv"

#today = datetime.today().date()

##@st.cache_data
#def data(file):
#    df = pd.read_csv(file, delimiter=";", skipinitialspace = True)
#    return df


##@st.cache_data
#def dataNewversion(file):
#    df = pd.read_csv(file, delimiter=",", skipinitialspace = True)
#    return df



##@st.cache_data
#def dataNew(file):
#    df = pd.read_csv(file, delimiter=",", skipinitialspace = True, index_col="Date", parse_dates=True, dayfirst=True)
#    return df


#@st.cache_data
def dataNew(file):
    df = pd.read_csv(file, delimiter=",", skipinitialspace = True)
    #df['Date'] = df.apply(lambda x: datetime.strptime(x['Date'], "%Y/%m/%d").date(), axis=1)
    df['Date'] = df.apply(lambda x: datetime.strptime(x['Date'], "%d/%m/%Y").date(), axis=1)
    df.index = df['Date']
    df = df.drop(['Date'], axis=1, errors='ignore')
    return df


#@st.cache_data
#def dataExcel(file):
#    df = pd.read_excel(file)
#    return df


# Plot the retruns for each stock
def monthlyReturns(returns, managers):
    fig = go.Figure()

    for manager, col in enumerate(returns.columns.tolist()):
        fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns[col],
            name=managers[manager]
        ))
        
    fig.update_layout(
        title={
            'text': "Returns",
        },
        template='seaborn',
        xaxis=dict(autorange=True,
                title_text='Date'),
        yaxis=dict(autorange=True,
                title_text='Returns'),
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.4,
        xanchor="left",
        x=0)
    )

    return fig


# Plot a Boxplot of each stocks returns
def boxReturns(returns, managers):
    fig = go.Figure()

    for manager, col in enumerate(returns.columns.tolist()):
        fig.add_trace(
        go.Box(
            y=returns[col],
            name=managers[manager]
        ))


    fig.update_layout(
        title={
            'text': "Box Plot",
        },
        template='seaborn',
        xaxis=dict(autorange=True,
                title_text='Sector'),
        yaxis=dict(autorange=True,
                title_text='Box Plot Distribution'),
        showlegend=False
        #legend=dict(
        #orientation="h",
        #yanchor="bottom",
        #y=1,
        #xanchor="left",
        #x=0)
    )

    return fig


# Plot the histogram of the each stocks returns
def histogramReturns(returns, managers):
    fig = go.Figure()

    for manager, col in enumerate(returns.columns.tolist()):
        fig.add_trace(
        go.Histogram(
            x=returns[col],
            name=managers[manager]
        ))


    fig.update_layout(
        title={
            'text': "Histogram",
        },
        template="seaborn",
        xaxis=dict(autorange=True,
                title_text='Daily Returns'),
        yaxis=dict(autorange=True,
                title_text='Count'),
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.4,
        xanchor="left",
        x=0) 
    )

    return fig




# Calculate cumulative returns of all portfolios
def cumReturns(returns, managers):
# Calculate the cumulative returns using the 'cumprod()' function
    cumulative_returns = (1 + returns).cumprod()

    # Plot the stocks cumulative returns
    fig = go.Figure()

    for manager, col in enumerate(cumulative_returns.columns.tolist()):
        fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[col],
            name=managers[manager]
        ))


    fig.update_layout(
        title={
            'text': "Cumulative Returns",
        },
        template="seaborn",
        xaxis=dict(autorange=True,
                title_text='Date'),
        yaxis=dict(autorange=True,
                title_text='Cumulative Return'),
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.4,
        xanchor="left",
        x=0)
    )

    return fig




# Plot the Manager Sharpe Ratios
def sharpeRatio(risk_return):
    fig = go.Figure()

    fig.add_trace(
    go.Bar(
        x=risk_return.index,
        y=risk_return['Sharpe'],
        text=risk_return['Sharpe'],
        marker={'color': 'teal'}
    ))


    fig.update_layout(
        title={
            'text': "Sharpe Ratios",
        },
        template="seaborn",
        xaxis=dict(autorange=True,
                title_text='Manager'),
        yaxis=dict(autorange=True,
                title_text='Ratio')
    )

    return fig



def scatterPlot(risk_return):

    fig = px.scatter(risk_return, x='Volatility', y='Returns', color='Volatility', title='Risk vs. Return Profile', 
                     text=risk_return.index, 
                     color_continuous_scale=px.colors.sequential.Aggrnyl_r,
                     #size_max=30
                )

    fig.update_layout(
        template='seaborn',
        font=dict(size=10),
        yaxis_tickformat = '.1%',
        xaxis_tickformat = '.1%',
        #hoverlabel=dict(
        #bgcolor="white",
        #font_size=8,
        #font_family="Rockwell"),
    )

    fig.update_traces(hovertemplate='Volatility: %{x} <br>Returns: %{y}')
    fig.update_traces(marker={'size': 15})
    
    return fig

def correlationGraph(correlation):
    fig = px.imshow(correlation, color_continuous_scale=px.colors.sequential.Aggrnyl_r, text_auto=True, title='Correlation Matrix')
    fig.update_layout(width=700,height=700)
    return fig



def efficient(returns, simulations):

    # Setup lists to hold portfolio weights, returns and volatility

    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights

    num_assets = len(returns.columns)
    num_portfolios = simulations

    cov_matrix = returns.apply(lambda x: np.log(1+x)).cov()

    # Calculate Portfolio weights for num_portfolios

    for portfolio in range(num_portfolios):
        weights = np.random.rand(num_assets)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returnsPortfolio = np.dot(weights, annual_returns) # Returns are the product of individual expected returns of asset and its # weights 
        p_ret.append(returnsPortfolio)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(12) # Annual standard deviation = volatility
        p_vol.append(ann_sd)



    # Insert the stock weights that correspond to the respective portfolio return and volatility

    data = {'Returns':p_ret, 'Volatility':p_vol}

    for counter, symbol in enumerate(returns.columns.tolist()):
        data[symbol+' weight'] = [w[counter] for w in p_weights]

    
    # Create portfolios dataframe to hold the portfolio weights of stocks, and portfolio return, volatility and sharpe ratio

    portfolios  = pd.DataFrame(data)

    portfolios['Sharpe'] = (portfolios['Returns']-0.05)/portfolios['Volatility']

    return portfolios



def optimalPortfolio(portfolios, minReturn):

    # Finding the optimal portfolio
    risky_port = portfolios.loc[((portfolios['Returns']) >= minReturn)]
    #risky_port = portfolios.loc[(portfolios['Volatility']) >= minVol]
    risky_port = risky_port.reset_index(drop=True)
    optimal_risky_port = risky_port.iloc[(risky_port['Sharpe']).idxmax()]

    return optimal_risky_port



def optimalTable(opt_port_df):
    # Optimal Portfolio Stock Weights Table - Sample Data

    head = ['<b>Symbol<b>', '<b>Weight<b>']

    labels =[]
    colorBreak = []

    for val, manager in enumerate(opt_port_df['Managers'].tolist()):
        txt='<b>'+manager+'<b>'
        labels.append(txt)


        if((val % 2) == 0):

            colorBreak.append('white')
        else:
            colorBreak.append('#b8df10')

        
    wght =[]
        
    for weight in opt_port_df['Weight'].tolist():
        w = '{:.2%}'.format(weight)
        wght.append(w)

        
    fig = go.Figure(data=[go.Table(
        header=dict(values=head,
                    fill_color='darkolivegreen',
                    line_color='darkslategray',
                    font=dict(color='white', size=10),
                    align=['left', 'center']),
        cells=dict(values=[labels, wght],
                fill_color=[colorBreak],
                line_color='darkslategray',
                font=dict(color='black', size=10),
                align=['left','center']))
    ])

    fig.update_layout(height=175, width=400, margin=dict(l=0, r=0, b=0,t=0))

    return fig


def efficientFrontier(portfolios, optimal_port_df):
    # Plot efficient frontier with optimal portfolio

    fig = px.scatter(portfolios, x='Volatility', y='Returns', color='Sharpe', title='Portfolio Efficient Frontier with Optimal Portfolio',
                     color_continuous_scale=px.colors.sequential.Aggrnyl_r)
    fig.add_trace(go.Scatter(x=optimal_port_df['Volatility'], y=optimal_port_df['Returns'], name='Optimal Portfolio',
                marker=dict(
                color='LightSkyBlue',
                size=20,
                )))

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.1,
        xanchor="right",
        x=1
        ),
        template='seaborn',
        yaxis_tickformat = '.2%',
        xaxis_tickformat= '.2%',
        width=700,
        height=700)
    
    return fig



def portfolioMetrics(returns, optimal_risky_port):
    # Calculate Historical Optimal Portfolio Return using optimal weights

    optweights = optimal_risky_port[2:-1].values

    portfolio_returns = returns.dot(optweights)    

    # Convert the historical optimal portfolio returns to a dataframe

    port_returns = pd.DataFrame(portfolio_returns)
    port_returns.columns = ['Portfolio_returns']

    # Calculate the historical cumulative returns for the optimal portfolio

    optimal_cumulative_returns = (1 + port_returns).cumprod()
 

    return port_returns, optimal_cumulative_returns



def descriptiveStatistics(startDate, endDate, initial, optimal_cumulative_returns, port_returns, optimal_risky_port):


    # Descriptive Statistics - Sample Data

    start = str(startDate.day)+'-'+str(startDate.month)+'-'+str(startDate.year)
    end = str(endDate.day)+'-'+str(endDate.month)+'-'+str(endDate.year)

    months = port_returns.shape[0]

    init_investment = initial

    pm_start = init_investment

    pm_end = round(init_investment * optimal_cumulative_returns['Portfolio_returns'][-1],2)

    returnsPos = port_returns.Portfolio_returns[(port_returns['Portfolio_returns'] > 0)].count()
    returnPosPerc = round((returnsPos/months),6)

    returnPosSum = port_returns.Portfolio_returns[(port_returns['Portfolio_returns'] > 0)].sum()
    returnPosAvg = round((returnPosSum/returnsPos),6)

    returnsNeg = port_returns.Portfolio_returns[(port_returns['Portfolio_returns'] <= 0)].count()
    returnNegPerc = round((returnsNeg/months),6)

    returnNegSum = port_returns.Portfolio_returns[(port_returns['Portfolio_returns'] <= 0)].sum()
    returnNegAvg = round((returnNegSum/returnsNeg),6)

    monthly_pm_max_return = port_returns['Portfolio_returns'].max()

    monthly_pm_min_return = port_returns['Portfolio_returns'].min()

    monthly_pm_mean_return = port_returns['Portfolio_returns'].mean()

    monthly_pm_median_return = port_returns['Portfolio_returns'].median()

    pm_return = optimal_risky_port[0]

    pm_vol = optimal_risky_port[1]
    
    optweights = optimal_risky_port[2:-1].values
    pm_md = round((np.dot(optweights, annual_std_dev))*100,2)

    pm_sharpe = round(optimal_risky_port[-1],2)


    return start, end, months, pm_start, pm_end, returnsPos, returnPosPerc, returnPosAvg, returnsNeg, returnNegPerc, returnNegAvg, monthly_pm_max_return, monthly_pm_min_return, monthly_pm_mean_return, monthly_pm_median_return, pm_return, pm_vol, pm_md, pm_sharpe





def descriptiveTable(start, end, months, pm_start, pm_end, returnsPos, returnPosPerc, returnPosAvg, returnsNeg, returnNegPerc, returnNegAvg, monthly_pm_max_return, monthly_pm_min_return, monthly_pm_mean_return, monthly_pm_median_return, pm_return, pm_vol, pm_md, pm_sharpe):
    # Table of Descriptive Statistics - Sample Data

    colorScheme=['white', '#b8df10']

    head = ['<b>Statistic<b>', '<b>Optimal Portfolio<b>']
    labels = ['<b>Start Date<b>', '<b>End Date<b>', '<b>Starting Investment<b>', '<b>Ending Investment<b>', '--------------------------------------',
            '<b>Months<b>', '<b>Positive Months<b>', '<b>% Positive<b>', '<b>Negative Months<b>', '<b>% Negative<b>',
            '<b>Average Profit<b>', '<b>Average Loss<b>', '--------------------------------------','<b>Max<b>', '<b>Min<b>', '<b>Mean<b>', '<b>Median<b>', 
            '<b>Annualized Return<b>', '<b>Annualized Volatility<b>', '<b>Portfolio MD<b>', '<b>Sharpe Ratio<b>']
    pf_stats = [start, end, '${:,}'.format(pm_start), '${:,}'.format(pm_end), '--------------------------------------', months, returnsPos, '{:.2%}'.format(returnPosPerc), 
                returnsNeg, '{:.2%}'.format(returnNegPerc), '{:.2%}'.format(returnPosAvg),
                '{:.2%}'.format(returnNegAvg), '--------------------------------------','{:.2%}'.format(monthly_pm_max_return), 
                '{:.2%}'.format(monthly_pm_min_return), '{:.2%}'.format(monthly_pm_mean_return),
                '{:.2%}'.format(monthly_pm_median_return), '{:.2%}'.format(pm_return), '{:.2%}'.format(pm_vol), pm_md, pm_sharpe]


    fig = go.Figure(data=[go.Table(
        header=dict(values=head,
                    fill_color='darkolivegreen',
                    line_color='darkslategray',
                    font=dict(color='white', size=10),
                    align=['left','center']),
        cells=dict(values=[labels, pf_stats],
                fill_color=[colorScheme*11],
                font=dict(color='black', size=10),
                line_color='darkslategray',
                align=['left','center']))
    ])

    fig.update_layout(margin=dict(l=0, r=0, b=0,t=0),  width=400, height=675)

    return fig




def monteCarloTable(managerMC, lowerPercentile, upperPercentile):

    head = ['<b>Managers<b>', '<b>{} Rts Lower<b>'.format(lowerPercentile), '<b>{}Rts Upper<b>'.format(upperPercentile), '<b>{} Invest Lower<b>'.format(lowerPercentile), '<b>{} Invest Upper<b>'.format(upperPercentile)]

    count = managerMC.shape[0]

    labels =[]
    colorBreak = []

    for val, manager in enumerate(managerMC['Managers'].tolist()):
        txt='<b>'+manager+'<b>'
        labels.append(txt)


        if((val % 2) == 0):

            colorBreak.append('white')
        else:
            colorBreak.append('#b8df10')

    lower = managerMC['lowerReturn'].tolist()
    upper = managerMC['upperReturn'].tolist()

    lowerInv = managerMC['lowerInvest'].tolist()
    upperInv = managerMC['upperInvest'].tolist()

    #pf_stats = [lower, upper, lowerInv, upperInv]


    fig = go.Figure(data=[go.Table(
    header=dict(values=head,
                fill_color='darkolivegreen',
                line_color='darkslategray',
                font=dict(color='white', size=10),
                align=['left','center','center','center','center']),
    cells=dict(values=[labels, lower, upper, lowerInv, upperInv],
            fill_color=[colorBreak],
            font=dict(color='black', size=10),
            line_color='darkslategray',
            format=['','.2%', '.2%', ',', ','],
            align=['left','center', 'center', 'center', 'center']))
    ])

    fig.update_layout(margin=dict(l=0, r=0, b=0,t=0),  width=600, height=count*50)

    return fig



def managerDescription(startDate, endDate, initial, returns, managerList, yrs_full):

    count = len(managerList)

    allManagerDescription = []

    alignManagers = ['left']


    for manager in managerList:

        managerDF = returns[manager]

        #st.dataframe(managerDF)

        cumReturns = (1 + managerDF).cumprod()
        annReturns = managerDF.apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs_full) - 1
        #ann_std_dev = managerDF.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(12))
        ann_std_dev_a = managerDF.apply(lambda x: np.log(1+x)).std()
        ann_std_dev = ann_std_dev_a * np.sqrt(12)

        ann_sharpe = round(annReturns/ann_std_dev,2)

        #risk_return = pd.concat([annReturns, ann_std_dev], axis=1) # Creating a table for visualising returns and volatility of assets
        #risk_return.columns = ['Returns', 'Volatility']

        #risk_return['Sharpe'] = risk_return.apply(lambda x: round(x['Returns']/x['Volatility'],2), axis=1)

        #st.dataframe(cumReturns)
        #st.dataframe(managerDF)

        # Descriptive Statistics - Sample Data

        start = str(startDate.day)+'-'+str(startDate.month)+'-'+str(startDate.year)
        end = str(endDate.day)+'-'+str(endDate.month)+'-'+str(endDate.year)

        months = returns.shape[0]

        init_investment = initial

        pm_start = init_investment

        #pm_end = round(init_investment * cumReturns[manager][-1],2)
        pm_end = round(init_investment * cumReturns[-1],2)


###############

        returnsPos = managerDF[(managerDF > 0)].count()
        returnPosPerc = round((returnsPos/months),6)

        returnPosSum = managerDF[(managerDF > 0)].sum()
        returnPosAvg = round((returnPosSum/returnsPos),6)

        returnsNeg = managerDF[(managerDF <= 0)].count()
        returnNegPerc = round((returnsNeg/months),6)

        returnNegSum = managerDF[(managerDF <= 0)].sum()
        returnNegAvg = round((returnNegSum/returnsNeg),6)

        monthly_pm_max_return = managerDF.max()

        monthly_pm_min_return = managerDF.min()

        monthly_pm_mean_return = managerDF.mean()

        monthly_pm_median_return = managerDF.median()

        pm_sharpe = ann_sharpe

        pm_return = annReturns

        pm_vol = ann_std_dev

        #pm_md = pm_sharpe

        pf_stats = [start, end, '${:,}'.format(pm_start), '${:,}'.format(pm_end), '--------------------------------------', months, returnsPos, '{:.2%}'.format(returnPosPerc), 
                returnsNeg, '{:.2%}'.format(returnNegPerc), '{:.2%}'.format(returnPosAvg),
                '{:.2%}'.format(returnNegAvg), '--------------------------------------','{:.2%}'.format(monthly_pm_max_return), 
                '{:.2%}'.format(monthly_pm_min_return), '{:.2%}'.format(monthly_pm_mean_return),
                '{:.2%}'.format(monthly_pm_median_return), '{:.2%}'.format(pm_return), '{:.2%}'.format(pm_vol), pm_sharpe]
        

        allManagerDescription.append(pf_stats)
        alignManagers.append('center')

        colorScheme=['white', '#b8df10']


##############

    head = ['<b>Statistic<b>']

    #for count, manager in enumerate(managerList):
    for manager in managerList:

        head.append('<b>'+manager+'<b>')
    
    
    
    
    labels = ['<b>Start Date<b>', '<b>End Date<b>', '<b>Starting Investment<b>', '<b>Ending Investment<b>', '--------------------------------------',
            '<b>Months<b>', '<b>Positive Months<b>', '<b>% Positive<b>', '<b>Negative Months<b>', '<b>% Negative<b>',
            '<b>Average Profit<b>', '<b>Average Loss<b>', '--------------------------------------','<b>Max<b>', '<b>Min<b>', '<b>Mean<b>', '<b>Median<b>', 
            '<b>Annualized Return<b>', '<b>Annualized Volatility<b>', '<b>Sharpe Ratio<b>']
    
    
    allManagerDescription.insert(0, labels)

    #pf_stats = [start, end, '${:,}'.format(pm_start), '${:,}'.format(pm_end), '--------------------------------------', months, returnsPos, '{:.2%}'.format(returnPosPerc), 
    #            returnsNeg, '{:.2%}'.format(returnNegPerc), '{:.2%}'.format(returnPosAvg),
    #            '{:.2%}'.format(returnNegAvg), '--------------------------------------','{:.2%}'.format(monthly_pm_max_return), 
    #            '{:.2%}'.format(monthly_pm_min_return), '{:.2%}'.format(monthly_pm_mean_return),
    #            '{:.2%}'.format(monthly_pm_median_return), '{:.2%}'.format(pm_return), '{:.2%}'.format(pm_vol), pm_md, pm_sharpe]


    fig = go.Figure(data=[go.Table(
        header=dict(values=head,
                    #fill_color='#390879',
                    fill_color='darkolivegreen',
                    align=alignManagers,
                    line_color='darkslategray',
                    font=dict(color='white', size=10)),
        #cells=dict(values=[labels, pf_stats],
        cells=dict(values=allManagerDescription,
                line_color='darkslategray',
                #fill_color='lavender',
                fill_color=[colorScheme*10],
                font=dict(color='black', size=10),
                align=alignManagers))
    ])

    #fig.update_layout(margin=dict(l=0, r=0, b=0,t=0),  width=(201*count), height=650)
    fig.update_layout(margin=dict(l=0, r=0, b=0,t=0),  width=600, height=count*130)

    return fig


################


#def monte(numSim, simYears, returns, optimal_risky_port):
#
#    optweights = optimal_risky_port[2:-1].values
#
#    MC_year = MCSimulation(
#        portfolio_data = returns,
#        weights = optweights.tolist(),
#        num_simulation = numSim,
#        num_trading_months = 12*simYears,
#        distributionDD = 0)
#
#    return MC_year


#with st.form("my-form", clear_on_submit=True):
#    fileUpload = st.file_uploader("FILE UPLOADER")
#    submitted = st.form_submit_button("UPLOAD!")
#
#    if submitted and fileUpload is not None:
#
#        dataframe = dataNew(fileUpload)

st.markdown("<h1 style='text-align: left; color: teal; padding-left: 0px; font-size: 40px'><b>Manager Analysis<b></h1>", unsafe_allow_html=True)
st.markdown(" ")

fileUpload = st.file_uploader("File upload")

try:

    dataframe = dataNew(fileUpload)

    returns = dataframe.dropna()

    managers = returns.columns.tolist()

    #managers.sort()

    managerList = st.multiselect('Select Constituents', managers, managers)

    #managerList.sort()

    returns = returns[managerList]

    date0 = pd.to_datetime(returns.index[0], format='%Y-%m-%d')
    date1 = pd.to_datetime(returns.index[-1], format='%Y-%m-%d')




    #url = 'animation_ljzfr9ug.json'
    url='data.json'

    with open(url, 'r') as fson:  
        res = json.load(fson)


    url_json = res


    with st.sidebar:
        st_lottie(url_json,
                # change the direction of our animation
                reverse=True,
                # height and width of animation
                height=200,  
                width=200,
                # speed of animation
                speed=1,  
                # means the animation will run forever like a gif, and not as a still image
                loop=True,  
                # quality of elements used in the animation, other values are "low" and "medium"
                quality='high',
                # THis is just to uniquely identify the animation
                key='Car',
                )


    #st.sidebar.header('Economic Activity')
    st.sidebar.markdown("<h1 style='text-align: left; color: coral; padding-left: 0px; font-size: 25px'><b>Anaysis Inputs<b></h1>", unsafe_allow_html=True)


    st.markdown("<h1 style='text-align: left; color: #872657; padding-left: 0px; font-size: 40px'><b>Snapsot<b></h1>", unsafe_allow_html=True)

    startDate = st.sidebar.date_input('Start Date', date0)
    endDate = st.sidebar.date_input('End Date', date1)

    yrs_full = ((endDate-startDate).days)/365

    returnTarget = st.sidebar.number_input('Target Return', min_value=0.0000, max_value=0.2000, value=0.0000, step=0.0025)
    #volTarget = st.number_input('Target Volatility', min_value=0.0000, max_value=0.2000, value=0.0000, step=0.0025)

    #numSim = st.number_input('Simulation Paths', min_value=100, max_value=1000, value=500, step=1)
    #simYears = st.number_input('Simulation Years', min_value=1, max_value=10, value=5, step=1)


    returns = returns[(returns.index >= startDate) & (returns.index <= endDate)]

    annual_returns = returns.apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs_full) - 1
    annual_std_dev = returns.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(12))
    correlation = returns.apply(lambda x: np.log(1+x)).corr()
    risk_return = pd.concat([annual_returns, annual_std_dev], axis=1) # Creating a table for visualising returns and volatility of assets
    risk_return.columns = ['Returns', 'Volatility']



    risk_return['Sharpe'] = risk_return.apply(lambda x: round(x['Returns']/x['Volatility'],2), axis=1)

    #for manager in managerList:
    #
    #    st.write(manager)
    #    st.dataframe(returns[manager])

    #managerDF = returns['Ci Diversified Income B']
    #cumuReturns = (1 + managerDF).cumprod() 

    #st.dataframe(cumuReturns)

    #st.write(returns)

    #monthly = monthlyReturns(returns, managers)
    monthly = monthlyReturns(returns, managerList)
    ##monthly

    #box = boxReturns(returns, managers)
    box = boxReturns(returns, managerList)
    ##box

    #histogram = histogramReturns(returns, managers)
    histogram = histogramReturns(returns, managerList)
    ##histogram

    #cumulative = cumReturns(returns, managers)
    cumulative = cumReturns(returns, managerList)
    ##cumulative


    col1, col2 = st.columns([1,1])

    col1.plotly_chart(monthly)
    col2.plotly_chart(cumulative)

    col3, col4 = st.columns([1,1])
    col3.plotly_chart(box)
    col4.plotly_chart(histogram)


    st.markdown("<h1 style='text-align: left; color: #872657; padding-left: 0px; font-size: 40px'><b>Risk Analysis<b></h1>", unsafe_allow_html=True)

    sharpe = sharpeRatio(risk_return)
    #sharpe

    scatter = scatterPlot(risk_return)
    #scatter

    col5, col6 = st.columns([1,1])
    col5.plotly_chart(sharpe)
    col6.plotly_chart(scatter)


    correlationG = correlationGraph(correlation.round(2))
    #correlationG

    managerStats = managerDescription(startDate, endDate, 10000, returns, managerList, yrs_full)
    #managerStats


    col7, col8 = st.columns([1,1])

    col7.plotly_chart(correlationG)



    #randomPortfolios = st.sidebar.number_input('Simulation Paths', min_value=1000, max_value=100000, value=10000, step=1000)
    randomPortfolios = 3600

    portfolios = efficient(returns, randomPortfolios)
    optimal_risky_port = optimalPortfolio(portfolios, returnTarget)

    opt_port_df = pd.DataFrame(data={'Managers': returns.columns.tolist(), 'Weight': optimal_risky_port[2:-1].values})
    optimal_port_df = pd.DataFrame(data={'Returns': optimal_risky_port[0], 'Volatility': optimal_risky_port[1]}, index=[0])


    frontierEfficient = efficientFrontier(portfolios, optimal_port_df)
    #frontierEfficient

    col8.plotly_chart(frontierEfficient)

    st.markdown("<h1 style='text-align: left; color: #872657; padding-left: 0px; font-size: 40px'><b>Manager/Asset Analysis<b></h1>", unsafe_allow_html=True)

    col9, col10 = st.columns([1, 1])
    
    #st.markdown("<h1 style='text-align: left; color: #872657; padding-left: 0px; font-size: 40px'><b>Optimal Portfolio Analysis<b></h1>", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: left; color: #872657; padding-left: 0px; font-size: 40px'><b>Optimal Portfolio Analysis<b></h1>", unsafe_allow_html=True)

    col9.markdown("<h3 style='text-align: left; color: black; padding-left: 0px; font-size: 17px'><b>Manager Stats<b></h3>", unsafe_allow_html=True)

    col9.plotly_chart(managerStats)

    ####col10 is for table statistic graphs

    col11, col12 = st.columns([1,1.5])

    optTable = optimalTable(opt_port_df)
    #st.markdown('Optimal Portfolio Weights')
    col11.markdown("<h1 style='text-align: left; color: black; padding-left: 0px; font-size: 20px'><b>Optimal Portfolio Weights<b></h1>", unsafe_allow_html=True)
    col11.plotly_chart(optTable)

    port_returns, optimal_cumulative_returns = portfolioMetrics(returns, optimal_risky_port)

    start, end, months, pm_start, pm_end, returnsPos, returnPosPerc, returnPosAvg, returnsNeg, returnNegPerc, returnNegAvg, monthly_pm_max_return, monthly_pm_min_return, monthly_pm_mean_return, monthly_pm_median_return, pm_return, pm_vol, pm_md, pm_sharpe = descriptiveStatistics(startDate, endDate, 10000, optimal_cumulative_returns, port_returns, optimal_risky_port)


    descriptiveTableFig = descriptiveTable(start, end, months, pm_start, pm_end, returnsPos, returnPosPerc, returnPosAvg, returnsNeg, returnNegPerc, returnNegAvg, monthly_pm_max_return, monthly_pm_min_return, monthly_pm_mean_return, monthly_pm_median_return, pm_return, pm_vol, pm_md, pm_sharpe)
    #st.markdown('Optimal Portfolio Statistics')
    col11.markdown("<h1 style='text-align: left; color: black; padding-left: 0px; font-size: 20px'><b>Optimal Portfolio Statistics<b></h1>", unsafe_allow_html=True)
    col11.plotly_chart(descriptiveTableFig)



    st.sidebar.markdown("<h1 style='text-align: left; color: coral; padding-left: 0px; font-size: 25px'><b>MC Simulation<b></h1>", unsafe_allow_html=True)

    #numSim = st.sidebar.number_input('Simulation Paths', min_value=100, max_value=1000, value=500, step=1)
    numSim = 360
    simYears = st.sidebar.number_input('Simulation Years', min_value=1, max_value=10, value=5, step=1)
    drawDown = st.sidebar.number_input('Yearly Drawdown', min_value=0.00, max_value=0.20, value=0.00, step=0.01)


    values = st.sidebar.slider(
        'Select Percentile Range',
        min_value=0.0, max_value=1.0, value=(0.025, 0.975), step=0.025)
    

    col10.markdown("<h1 style='text-align: left; color: black; padding-left: 0px; font-size: 20px'><b>MC Simulation - Single Manager/Asset<b></h1>", unsafe_allow_html=True)
    #col10.write(" ")
    #col10.write(" ")
    managerSelect = col10.selectbox('Single Manager/Asset', managerList)

    indexVal = managerList.index(managerSelect)


    confidence = round(((values[1] - values[0])*100),2)
    confidenceInterval = str(confidence)+'%'
    confidenceLower = str(round(values[0]*100,2))+'%'
    confidenceUpper = str(round(values[1]*100,2))+'%'
  

    ###############

    lowerReturn = []
    upperReturn =[]
    lowerInvest = []
    upperInvest = []

    simulationGraphs = []
    distributionGraphs = []

    for manager in managerList:

        #managerSim =pd.DataFrame(data=returns[manager].values)
        managerSim = returns[[manager]]

        #st.dataframe(returns)

        #st.dataframe(managerSim)

        MC_yearSim = MCSimulation(
        portfolio_data = managerSim,
        weights = [1],
        num_simulation = numSim,
        num_trading_months = 12*simYears,
        distributionDD = drawDown,
        lower=values[0],
        upper=values[1])

        #st.plotly_chart(MC_yearSim.plot_simulation())

        simulationGraphs.append(MC_yearSim.plot_simulation())

        #st.plotly_chart(MC_yearSim.plot_distribution())

        distributionGraphs.append(MC_yearSim.plot_distribution())

        statisticsSim, intervalsSim = MC_yearSim.summarize_cumulative_return()

        #st.write(statistics)
        #st.write(intervals)


        # calculate upper and lower bounds percentile returns

        return_lowerSim = round(((intervalsSim[0])**(1/simYears))-1,6)
        return_upperSim = round(((intervalsSim[1])**(1/simYears))-1,6)

    #    # Print results
    #    st.write("There is a 95% chance that the anuualized return of the portfolio"
    #        " over the next {:,} years will be within in the range of"
    #        " {:.2%} and {:.2%}".format(simYears, return_lowerSim, return_upperSim))

        lowerReturn.append(return_lowerSim)
        upperReturn.append(return_upperSim)

        # Set initial investment
        initial_investment = 10000

        # Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $10,000
        ci_lowerSim = round(intervalsSim[0]*initial_investment,2)
        ci_upperSim = round(intervalsSim[1]*initial_investment,2)

        lowerInvest.append(ci_lowerSim)
        upperInvest.append(ci_upperSim)


    
    #col10.markdown("<h1 style='text-align: left; color: black; padding-left: 0px; font-size: 20px'><b>MC Simulation<b></h1>", unsafe_allow_html=True)
    #872657
    #col10.write(" ")
    col10.plotly_chart(simulationGraphs[indexVal])
    col10.plotly_chart(distributionGraphs[indexVal])


    col10.markdown("<h1 style='text-align: left; color: black; padding-left: 0px; font-size: 20px'><b>MC Results - Single Manager/Asset<b></h1>", unsafe_allow_html=True)
    col10.write(" ")

    col10.write("There is a {} chance [{}-{} percentile] that the anuualized return of the portfolio"
        " over the next {:,} years will be within in the range of"
        " {:.2%} and {:.2%}".format(confidenceInterval, confidenceLower,confidenceUpper, simYears, lowerReturn[indexVal], upperReturn[indexVal]))



    col10.write("There is a {} chance [{}-{} percentile] that an initial investment of R{:0,.2f} in the portfolio"
        " over the next {:,} years will end within in the range of"
        " R{:0,.2f} and R{:0,.2f}".format(confidenceInterval, confidenceLower,confidenceUpper, initial_investment, simYears, lowerInvest[indexVal], upperInvest[indexVal]))


   



    #col10.write('Tester')

    returnsDictionary = {'Managers': managerList, 'lowerReturn': lowerReturn, 'upperReturn': upperReturn, 'lowerInvest': lowerInvest, 'upperInvest': upperInvest}

    managerMC = pd.DataFrame(data=returnsDictionary)
    #st.dataframe(managerMC)


    monteTable = monteCarloTable(managerMC, confidenceLower, confidenceUpper)
    col9.markdown("<h3 style='text-align: left; color: black; padding-left: 0px; font-size: 17px'><b>Manager Risk Analysis<b></h3>", unsafe_allow_html=True)
    col9.plotly_chart(monteTable)


    ###############

    optweights = optimal_risky_port[2:-1].values

    MC_year = MCSimulation(
    portfolio_data = returns,
    weights = optweights.tolist(),
    num_simulation = numSim,
    num_trading_months = 12*simYears,
    distributionDD = drawDown)

    col12.markdown("<h1 style='text-align: left; color: black; padding-left: 0px; font-size: 20px'><b>MC Simulation - Optimal Portfolio<b></h1>", unsafe_allow_html=True)

    col12.plotly_chart(MC_year.plot_simulation())

    col12.plotly_chart(MC_year.plot_distribution())

    statistics, intervals = MC_year.summarize_cumulative_return()

    #st.write(statistics)
    #st.write(intervals)


    # calculate upper and lower bounds percentile returns

    return_lower = round(((intervals[0])**(1/simYears))-1,6)
    return_upper = round(((intervals[1])**(1/simYears))-1,6)

    col12.markdown("<h1 style='text-align: left; color: black; padding-left: 0px; font-size: 20px'><b>MC Results - Optimal Portfolio<b></h1>", unsafe_allow_html=True)
    col12.write(" ")

    # Print results
    col12.write("There is a {} chance [{}-{} percentile] that the anuualized return of the portfolio"
        " over the next {:,} years will be within in the range of"
        " {:.2%} and {:.2%}".format(confidenceInterval, confidenceLower,confidenceUpper, simYears, return_lower, return_upper))


    # Set initial investment
    initial_investment = 10000

    # Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $10,000
    ci_lower = round(intervals[0]*initial_investment,2)
    ci_upper = round(intervals[1]*initial_investment,2)

    # Print results
    col12.write("There is a {} chance [{}-{} percentile] that an initial investment of R{:0,.2f} in the portfolio"
        " over the next {:,} years will end within in the range of"
        " R{:0,.2f} and R{:0,.2f}".format(confidenceInterval, confidenceLower,confidenceUpper, initial_investment, simYears, ci_lower, ci_upper))





    ############################################################################################


    #    # Print results
    #    st.write("There is a 95% chance that an initial investment of R{:0,.2f} in the portfolio"
    #        " over the next {:,} years will end within in the range of"
    #        " R{:0,.2f} and R{:0,.2f}".format(initial_investment, simYears, ci_lowerSim, ci_upperSim))





        ############################################################################################


except:
    st.write("Please upload file or check inputs!!")