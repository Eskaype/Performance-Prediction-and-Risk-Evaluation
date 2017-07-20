from bokeh.layouts import row, column
from bokeh.plotting import figure, output_server,show,output_file
from bokeh.palettes import RdYlGn4
from bokeh.models import CustomJS, ColumnDataSource,HoverTool,Legend,Range1d,LabelSet
from bokeh.client import push_session
from bokeh.models.widgets import PreText,TextInput,DataTable,TableColumn,PreText
from bokeh.io import curdoc,curstate
from operator import sub

import sys
import time
from math import pi

import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from yahoo_finance import Share
from datetime import datetime
from bokeh.models import DatetimeTickFormatter
from datetime import timedelta


def nix(val, lst):
    return [x for x in lst if x != val]
# Let source be the data I am using to plot.
curstate().autoadd = False
# Create a dictionary for table
source_table = ColumnDataSource(data=dict(Forecast = [],AQuarter = [], OneFourth = [],OneThird = [],TwoThird=[]))
columnsD = [
        TableColumn(field="Forecast", title="Days past Current"),
        TableColumn(field="AQuarter", title="96% Confidence"),
        TableColumn(field="OneFourth", title="80% Confidence"),
        TableColumn(field="OneThird", title="60% Confidence"),
    ]
data_table = DataTable(source=source_table, columns=columnsD, width=1050, height=300, row_headers=False, editable=True)

source_table2 = ColumnDataSource(data=dict(CurrentYield = []))
columnsE = [
        TableColumn(field="CurrentYield", title="Current Yield")
    ]
data_table2 = DataTable(source=source_table2, columns=columnsE, width=200, height=60, row_headers=False)


#output_file("Stock & Yield")
source_static1 = ColumnDataSource(data=dict(Yield_max = [],Yield_min = [],StockPrice = [],Mean_pred = []))
source_static2 = ColumnDataSource(data=dict(Yield = [],slope22 = []))
source_static3  = ColumnDataSource(data = dict(StockPrice = [], Time = []))
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN","GNL" , "GM","VZ","XOM"]# "KO", "GE","PG" , "VZ"]

datacoords = ColumnDataSource(data=dict(x=[], y1=[], y2=[],y3=[],names = ['15 DAYS','45 DAYS','60 DAYS']))
polyline =  ColumnDataSource(data=dict(a_x=[], b_x=[], a_y=[],b_y=[],m_x = [],m_y = []))
stocksource =ColumnDataSource(data=dict(s_x=[],s_y=[]))
global  New_Stock
global  good_Stock
New_Stock = True
good_Stock = True
window_size= [30 ,7]
step_size = [ 30 ,3 ]
days = [ 100 , 180]
# Start the output server session
output_server("Stock & Yield")
ticker1 = TextInput(value="IBM", title="Test Stock:",width = 200)
ticker2 = TextInput(value="None", title="Current Yield:",width = 5)
# set up plots
tools = 'pan,wheel_zoom,xbox_select,reset,click'
p1 = figure(plot_width=1200, plot_height=400, title="Yield vs Time in Days:" , x_axis_label = "Time in Days", y_axis_label = "Yield",toolbar_location="above" )
p1.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %B %Y"],
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"],
    )
p1.xaxis.major_label_orientation = pi/4
#p2 = figure(plot_width=1200, plot_height=400, title="Yield vs Time:" , x_axis_label = "Time", y_axis_label = "Yield", x_range=Range1d(start=0, end=700) )
p1.title.text = str('   Performance  Predictions On  Bounds')
p1.title.align = 'center'
p1.background_fill = "white"
p1.title.text_font = "times"
p1.title_text_font_size = "20pt"
p1.xaxis.axis_label_text_font_size = "15pt"
p1.yaxis.axis_label_text_font_size = "15pt"
stockline = p1.line("s_x","s_y",color = "black",source=stocksource,line_width=1)
linefita = p1.line("a_x","a_y",color = "Blue",source=polyline,line_dash = [3,3],line_width=3)
linefitb = p1.line("b_x","b_y",color = "red",source=polyline,line_dash = [3,3],line_width=3)
linefitm = p1.line("m_x","m_y",color = "green",source=polyline,line_dash = [3,3],line_width=3)
linea= p1.inverted_triangle("x","y1",size=10,color="Blue",source=datacoords)
lineb = p1.inverted_triangle("x","y2",size=10,color="red",source=datacoords)
linec = p1.diamond_cross("x","y3",size=10,color="green",source=datacoords)
labels = LabelSet(x='x', y='y3', text='names', level='glyph',
               source=datacoords, render_mode='canvas')
p1.add_layout(labels)
p1.grid
p1.ygrid.minor_grid_line_color = 'navy'
p1.ygrid.minor_grid_line_alpha = 0.1
legend = Legend(items=[
    ("Predicted Upper Bound"   , [linea]),
    ("Predicted Lower Bound" ,  [lineb]),
    ("Predicted Mean" ,  [linec]),
    ("Upper Bound"   , [linefita]),
    ("Lower Bound" ,  [linefitb]),
    ("Mean" ,  [linefitm]),
], location=(10, -50))

p1.add_layout(legend, 'right')


#ticker1= Select(title = "Options:",value="MSFT",
                           #options=["AAPL", "MSFT", "GOOG", "AMZN", "IBM", "HD", "WMT","XON","CUX","BAC","JPM","C"])



            

# New: put daily return value into OClist
def get_stock_daily_return_data(stock_name, total_days):
    end_date = str(datetime.now().date())
    #print end_date
    start_date = str((datetime.now() - timedelta(days=total_days)).date())
    #print start_date
    histdata = Share(stock_name).get_historical(start_date, end_date)
    #print len(histdata)
    OClist = []
    cnt =0
    temp =[]
    for i in range(len(histdata)):
        if((cnt%8)==0):
            temp =[]
            temp.append(float(histdata[i]['Open'])) 
            cnt = 0
        elif((cnt%7)==0):
            temp.append(float(histdata[i]['Close']))
            stock_return_value = (temp[1]-temp[0])/temp[0]
            OClist.append(stock_return_value)
                       
        cnt= cnt+1
    return OClist,end_date,start_date

# Each element in yield_list is the average value of window size
# Each element in risk_list is the std value of window size
def get_Confidence_list(predicted,current):
        gain_list = []
        Yield_list = []
        Days_list = [15,45,60]
        epsilon = [0.125,0.33333333333333,0.5,0.667]
        k=2
        while(k>=0):
            i=0
            Yield_list.append(Days_list[k])
            for i in range(len(epsilon)):
                 Yield = np.round((np.multiply(100,np.multiply((predicted[2][k] - predicted[3][k]),epsilon[i]) + predicted[5][k])), decimals = 4, out=None)
                 if(Yield > np.multiply(100,predicted[2][k])):
                     Yield = np.nan
                 #gain_list.append(Yield - current)
                 Yield_list.append(Yield)
                 
            k = k-1
        return gain_list, Yield_list
    
# Each element in yield_list is the average value of window size
# Each element in risk_list is the std value of window size
def get_predicted_maxmin(Maxmin , num1,num2,num3,K):
       PolyVal_Max = []
       PolyVal_Min = []
       PolyVal_Mean = []
       Pred_Min = []
       Pred_Max = []
       Pred_Mean = []
       #Weights = get_prediction_weights(len(Maxmin[0]))
       Coeff_mean=  np.polyfit(Maxmin[5],Maxmin[4],3)
       PolyVal_Mean = np.polyval(Coeff_mean, np.linspace(1,K,K));
       Coeff_max=  np.polyfit(Maxmin[2],map(sub,Maxmin[0],np.polyval(Coeff_mean,Maxmin[2]) ),4 )
       PolyVal_Max = np.polyval(Coeff_max, np.linspace(1,K,K))
       Coeff_min=  np.polyfit(Maxmin[3],map(sub,Maxmin[1],np.polyval(Coeff_mean,Maxmin[2])),4 )
       PolyVal_Min = np.polyval(Coeff_min, np.linspace(1,K,K));
       Pred_Mean = [np.polyval(Coeff_mean,K+num1)
                     ,np.polyval(Coeff_mean,K+num2),np.polyval(Coeff_mean,K+num3)]
       Pred_M = [np.polyval(Coeff_max,K+num1),np.polyval(Coeff_max,K+num2),np.polyval(Coeff_max,K+num3)]
       Pred_N = [np.polyval(Coeff_min,K+num1),np.polyval(Coeff_min,K+num2),np.polyval(Coeff_min,K+num3)]
       Pred_Max  = [x+y for x,y in zip(Pred_Mean , Pred_M)]
       Pred_Min = [x+y for x,y in zip(Pred_Mean , Pred_N)]
       

       
       return PolyVal_Max,PolyVal_Min,Pred_Max,Pred_Min,PolyVal_Mean,Pred_Mean

def get_max_min_values(Yield, sampling_interval):
        Yield_max = []
        Yield_min = []
        Mean_pred = []
        Time_max = []
        Time_min = []
        Time_mean = []
        i = 0;
        while(i<len(Yield)):
               window_start = i
               window_end = i+sampling_interval
               if(window_end > len(Yield)):
                   break
               computation_list = Yield[window_start:window_end]
               computation_array = np.asarray(computation_list)
               Mean_pred.append(np.mean(computation_array))
               Yield_max.append(np.max(computation_array))
               Yield_min.append(np.min(computation_array))
               Time_max.append(i+ np.argmax(computation_array))
               Time_min.append(i+ np.argmin(computation_array))
               Time_mean.append(i)
               i = i+2
        
        return Yield_max , Yield_min,Time_max,Time_min,Mean_pred,Time_mean

def update():
    result_list=[[],[],[],[],[],[]]
    predictedlist = [[],[],[],[],[],[]]
    confidence_list = [[],[]]
    Yearly_return_value_new = []
    data_plt = pd.DataFrame()
    
    Min = pd.DataFrame()
    Max = pd.DataFrame()
    Mean = pd.DataFrame()
    tabdata = pd.DataFrame()
    tabdata2 = pd.DataFrame()
    global N
    P =3
    tab = globals()['source_table']
    koo = globals()['data_table']
    tab2 = globals()['source_table2']
    roo = globals()['data_table2']
    global Yearly_return_value_new
    global result_list
    global New_Stock
    global predictedlist
    global confidence_list
    data = pd.DataFrame()
    # Update the Stock Price
    stock = ticker1.value
    [Yearly_return_value_new,end,start] = get_stock_daily_return_data(stock,1000) 
    #Max and Min Value
    N = len(Yearly_return_value_new)
    result_list = get_max_min_values(Yearly_return_value_new[1:N-P],2)
    #Prediction for max and min for 3 months
    predictedlist = get_predicted_maxmin(result_list,2,4,6,N-P)
    #confidence interval table
    confidence_list = get_Confidence_list(predictedlist,Yearly_return_value_new[N-P])
    stockline.data_source.data["s_x"] =  [datetime.now()-timedelta(days=N) + timedelta(days=i) for i in range(N-P)]
    stockline.data_source.data["s_y"] =  Yearly_return_value_new[1:N-P]
    linefita.data_source.data["a_x"] = [datetime.now()-timedelta(days=N) + timedelta(days=i) for i in (result_list[2])]
    linefita.data_source.data["a_y"] = result_list[0]
    linefitb.data_source.data["b_x"] = [datetime.now()-timedelta(days=N) + timedelta(days=i) for i in (result_list[3])]
    linefitb.data_source.data["b_y"] = result_list[1]
    linefitm.data_source.data["m_x"] = [datetime.now()-timedelta(days=N) + timedelta(days=i) for i in (result_list[5])]
    linefitm.data_source.data["m_y"] = result_list[4]
    linea.data_source.data["x"] = [datetime.now()-timedelta(days=N) + timedelta(days=N-P+15),datetime.now()-timedelta(days=N) + timedelta(days=N-P+30),datetime.now()-timedelta(days=N)+ timedelta(days=N-P+45)]
    linea.data_source.data["y1"] =predictedlist[2]
    lineb.data_source.data["x"] = [datetime.now()-timedelta(days=N) + timedelta(days=N-P+15),datetime.now()-timedelta(days=N) + timedelta(days=N-P+30),datetime.now()-timedelta(days=N) + timedelta(days=N-P+45)]
    lineb.data_source.data["y2"] =predictedlist[3]
    linec.data_source.data["x"] = [datetime.now()-timedelta(days=N) + timedelta(days=N-P+15),datetime.now()-timedelta(days=N) + timedelta(days=N-P+30),datetime.now()-timedelta(days=N) + timedelta(days=N-P+45)]
    linec.data_source.data["y3"] =predictedlist[4]
    tabdata = pd.DataFrame([confidence_list[1][0:5],confidence_list[1][5:10],confidence_list[1][10:15]],columns = ['Forecast','AQuarter','OneFourth','OneThird','TwoThird'] )
    koo.source.data = tab.from_df(tabdata)
        #source_table.data = tab.from_df(tabdata)
    tabdata2 = pd.DataFrame([np.round(np.multiply(Yearly_return_value_new[N-P],100),decimals = 4)], columns = ['CurrentYield'])
    roo.source.data = tab2.from_df(tabdata2)
    
def update_stats(data, t1):
    stats.text = str(data.describe())
    
def ticker1_change(attr, old, new):
    #ticker1.value = nix(new, DEFAULT_TICKERS)
    global New_Stock
    New_Stock = True
    update()

def ticker2_change(attr, old, new):
    #ticker1.value = nix(new, DEFAULT_TICKERS)
    global good_Stock
    good_Stock = True
    update()

def update_stats(data):
    stats.text = str(data.describe())
    
def selection_change(attrname, old, new):
    result_list=[[],[]]
    result_list_new = [[],[]]
    confidence_list = []
    global result_list
    global New_Stock
    data = pd.DataFrame()
   # Update the Stock Price
    stock = ticker1.value
    result_list=[[],[],[],[],[],[]]
    predictedlist = [[],[],[],[],[],[]]
    confidence_list = [[],[]]
    Yearly_return_value_new = []
    data_plt = pd.DataFrame()
    
    Min = pd.DataFrame()
    Max = pd.DataFrame()
    Mean = pd.DataFrame()
    tabdata = pd.DataFrame()
    tabdata2 = pd.DataFrame()
    global N
    P = 3
    foo =  globals()['source_static'+str(1)]
    tab = globals()['source_table']
    koo = globals()['data_table']
    tab2 = globals()['source_table2']
    roo = globals()['data_table2']
    global Yearly_return_value_new
    global result_list
    global New_Stock
    global predictedlist
    global confidence_list
    data = pd.DataFrame()
    # Update the Stock Price
    stock = ticker1.value
    Yearly_return_value_new = get_stock_daily_return_data(stock,1000) 
    #Max and Min Value
    N = len(Yearly_return_value_new)
    result_list = get_max_min_values(Yearly_return_value_new[1:N-P],4)
    #Prediction for max and min for 3 months
    predictedlist = get_predicted_maxmin(result_list,1,2,3,N-P)
    #confidence interval table
    confidence_list = get_Confidence_list(predictedlist,Yearly_return_value_new[N-P])
    stockline.data_source.data["s_x"] =  range(1,N-P)
    stockline.data_source.data["s_y"] =  Yearly_return_value_new[1:N-P]
    linefita.data_source.data["a_x"] = result_list[2]
    linefita.data_source.data["a_y"] = result_list[0]
    linefitb.data_source.data["b_x"] = result_list[3]
    linefitb.data_source.data["b_y"] = result_list[1]
    linefitm.data_source.data["m_x"] = result_list[5]
    linefitm.data_source.data["m_y"] = result_list[4]
    linea.data_source.data["x"] = [N-15+1,N-30+2 ,N-45+3]
    linea.data_source.data["y1"] =predictedlist[2]
    lineb.data_source.data["x"] = [N-15+1,N-30+2 ,N-45+3]
    lineb.data_source.data["y2"] =predictedlist[3]
    linec.data_source.data["x"] = [N-15+1,N-30+2 ,N-45+3]
    linec.data_source.data["y3"] =predictedlist[4]
    tabdata = pd.DataFrame([confidence_list[1][0:5],confidence_list[1][5:10],confidence_list[1][10:15]],columns = ['Forecast','AQuarter','OneFourth','OneThird','TwoThird'] )
    koo.source.data = tab.from_df(tabdata)
        #source_table.data = tab.from_df(tabdata)
    tabdata2 = pd.DataFrame([np.round(np.multiply(Yearly_return_value_new[N-P],100),decimals = 4)], columns = ['CurrentYield'])
    roo.source.data = tab2.from_df(tabdata2)

colors = ['blue', 'green', 'red', 'cyan']
ticker1.on_change("value", ticker1_change)
ticker2.on_change("value", ticker2_change)
source_static1.on_change('selected', selection_change)
source_static2.on_change('selected', selection_change)
source_static3.on_change('selected', selection_change)
source_table.on_change('selected',selection_change)

#Plot1 = row(p1,p2)
#Setting up web page layout
Plot2 = row(p1)
Plot3 = row(ticker1,data_table2)
Plot4 = column(data_table)
layout = column(Plot3,Plot2,Plot4)


#initialise

update()
curdoc().add_root(layout)
curdoc().title = "Stocks"
session = push_session(curdoc())
curdoc().add_periodic_callback(update(), 1000)
session.show()
session.loop_until_closed()


