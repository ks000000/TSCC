# -*- coding: utf-8 -*-
"""

The following functions use a simple dataframe, so you can understand what
the variables may look like in your own project.

Dataframe
---------

>>> d = {'col1': [1, 9, 3, 6, 20, 8, 3, 1, 30, 5]}
>>> df = pd.DataFrame(data=d)
	col1
0	1
1	9
2	3
3	6
4	20
5	8
6	3
7	1
8	30
9	5

"""
#from geojson import FeatureCollection, Feature, Polygon
#from folium import  Map, Choropleth
#import wradlib.ipol as ipol
import numpy as np
import pandas as pd


def getGrid(loc, r, n_steps = 50 ):

    r"""
    Receive a grid around your location.

    Parameters
    ----------
    loc : List of length two
        Coordinates to transform.
        For input use loc format like this [lat, lon].
    r : integer
        Radius from your location in [m].
    n_steps : number, optional
        Number of points you want to get.
        The default is 50.

    Returns
    -------
    array
        A number of coordinates???

    Examples
    --------
    >>> loc = [51.516666, 7.100000] # lat, lon of Gelsenkirchen
    >>> r = 20000
    >>> test = getGrid(loc, r, n_steps=50)
    array([[-19948.483334  , -19992.9       ],
           [-19132.15680339, -19992.9       ],
           [-18315.83027278, -19992.9       ],
           ...,
           [ 18418.86360478,  20007.1       ],
           [ 19235.19013539,  20007.1       ],
           [ 20051.516666  ,  20007.1       ]])
    """
        
    arr_grid = np.meshgrid(
            np.linspace( loc[0]-r, loc[0]+r,n_steps),
            np.linspace( loc[1]-r, loc[1]+r,n_steps))
    return np.vstack((arr_grid[0].ravel(),
                      arr_grid[1].ravel())).T

def getValueFromGrid(arr_grid, loc, arr_I):

    r"""
    Finding nearest station and get the ???

    Parameters
    ----------
    arr_grid : array from function getGrid()???
    loc : list of length two
        Coordinates to transform.
        For input use loc format like this [lat, lon].
    arr_I : ???    

    Returns
    -------
    float???

    Examples
    --------
    >>> ohne Beispiel?
    """

    return arr_I[ np.argmin([np.linalg.norm(x) for x in arr_grid-loc])]

def getInterpolation(arr_stations, arr_grid, s_method='idw', remove_missing = False):
    
    r"""
    Creates a modell for geostatical interpolation for the stations arr_stations in the grid arr_grid

    Parameters
    ----------
    arr_stations : numpy.array
        Contains utm coordinates of the stations.
    arr_grid : numpy.array
        Contains the grid in which the stations are in and to interpolate values for these points.
    s_method : str, optional
        Method to use for interpolation.
        The default is 'idw'.
    remove_missing : boolean, optional
        Remove missing values???
        The default is False.

    Returns
    -------
    modell : function
        Map the grid points depending on input values at the stations.

    Examples
    --------
    >>> ohne Beispiel?
    """
    
    if s_method == 'idw':
        modell = ipol.Idw(arr_stations, arr_grid, remove_missing = remove_missing)
    elif s_method =='nn':
        modell = ipol.Nearest(arr_stations, arr_grid, remove_missing = remove_missing)
    elif s_method =='linear':
        modell = ipol.Linear(arr_stations, arr_grid, remove_missing = remove_missing)
    elif s_method =='ok':
        modell = ipol.OrdinaryKriging(arr_stations, arr_grid, remove_missing = remove_missing)
    else:
        modell = lambda x:0
        print('please use idw, nn, linear or ok')
    return modell

def getGeoStatValue(loc, loc_stations, vals_at_stations, edges, method = 'idw', remove_missing = False):

    r"""
    Keine Ahnung.

    Parameters
    ----------
    loc : list of length two
        Coordinates to transform.
        For input use loc format like this [lat, lon].
    loc_stations : list???
        Location of some stations.???
    vals_at_stations : ???
    edges : array from function getGrid()???
    method : str, optional
        Method to use for interpolation.
        The default is 'idw'.
    remove_missing : boolean, optional
        Remove missing values???
        The default is False.

    Returns
    -------

    Examples
    --------
    >>> ohne Beispiel?
    """
    
    # e.g. idw = ipol.Idw(loc_stations, edges)#remove_missing = True)
    idw = getInterpolation(loc_stations, edges, s_method=method, remove_missing = remove_missing)

    # interpolation of values: idw_vals = idw(vals)
    return getValueFromGrid(edges, loc, idw(vals_at_stations))


def getPlot(loc_stations, mid_X, mid_Y, r, vals_at_stations, remove_missing = False):

    r"""
    Create a plot of your ???

    Paramaters
    ----------
    loc_stations : list???
        Location of some stations.???
    mid_X : ???
    mid_Y : ???
    r : integer
        Radius from your location in [m].
    vals_at_station : ???
    remove missing : boolean, optional
        Remove missing values???
        The default is False.

    Returns
    -------
    plot
        A plot of ???

    Examples
    --------
    >>> ohne Beispiel?
    """

    import matplotlib.pyplot as plt

    edges = np.asarray(getGrid([mid_X, mid_Y], r, n_steps=40))

    # Target coordinates
    x_boundaries = np.linspace(mid_X - r, mid_X + r, 40)
    y_boundaries = np.linspace(mid_Y - r, mid_Y + r, 40)

    idw = ipol.Idw(loc_stations, edges, remove_missing = remove_missing)

    # interpolation of values
    idw_vals = idw(vals_at_stations)

    # Subplot layout
    def gridplot(interpolated, title=""):
        pm = ax.pcolormesh(x_boundaries, y_boundaries,
                           interpolated.reshape((len(x_boundaries), len(y_boundaries))))
        plt.axis("tight")
        ax.scatter(loc_stations[:, 0], loc_stations[:, 1], facecolor="red", s=10, marker='o')
        plt.title(title)
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")

        return pm, plt

    # Plot results
    fig = plt.figure(figsize=(8, 7))
    plt.set_cmap('Blues')
    ax = fig.add_subplot(221, aspect="equal")
    pm, plt = gridplot(idw_vals, "IDW")
    plt.tight_layout()
    plt.colorbar(pm)
    pm.set_clim(0, )

    return plt

#
# ###############################
# ## keep only maybe
# ##############################
#
# #own imports
# from package_dataquality.TSCC.transform_data.helpful_functions import myProjUTM
# from package_dataquality.TSCC.visualisation.maps import addDictPoints2Map
#
#
#
# def createBoxes(p, r):
#     x,y = p
#     return [[x-r,y-r],[x+r,y-r],[x+r,y+r],[x-r,y+r],[x-r,y-r]]
#
#
# def getVerticesGrid(p,r_box):
#     L = createBoxes(p,r_box)
#     return [[y[1],y[0]] for y in [myProjUTM(x[0],x[1]) for x in L]]
#
#
# def doThis(arr_inter_vals, vals, name_method, arr_stations, arr_grid, n_steps, r, loc,
#            legend_name='', show_stations=True, L_station_names=[]):
#     '''
#     create a folium map for optical view of geostatitical interpolation
#
#     Parameters
#     ----------
#     V : TYPE
#         DESCRIPTION.
#     vals: TYPE
#         DESCRIPTION.
#     name_method : TYPE
#         DESCRIPTION.
#     arr_stations : TYPE
#         DESCRIPTION.
#     arr_grid : TYPE
#         DESCRIPTION.
#     n_steps : TYPE
#         DESCRIPTION.
#     r : TYPE
#         DESCRIPTION.
#     loc : TYPE
#         DESCRIPTION.
#     legend_name : TYPE, optional
#         DESCRIPTION. The default is ''.
#     show_stations : TYPE, optional
#         DESCRIPTION. The default is True.
#     L_station_names : TYPE, optional
#         DESCRIPTION. The default is [].
#
#     Returns
#     -------
#     None.
#
#     '''
#
#     names = [str(x) for x in range(len(arr_inter_vals))]
#     r_box = r/(n_steps)*(1+1/(n_steps-1))
#
#     #create list of features with polygons to view grid on map
#     L_feature =[Feature(
#                     geometry=Polygon([getVerticesGrid(arr_grid[k],r_box)]),
#                     properties={"name":names[k],
#                                 "val":round(arr_inter_vals[k],2)},
#                     id=names[k]
#                     ) for k in range(len(arr_grid))]
#
#
#     m = Map(location = myProjUTM(loc[0],loc[1]),
#             tiles='Cartodb positron',
#             zoom_start = 11,)
#
#     c = Choropleth(
#         geo_data=FeatureCollection(L_feature),
#         data=pd.DataFrame(list(zip(*[names, arr_inter_vals]))),
#         columns=[0, 1],
#         key_on="feature.id",
#         legend_name=legend_name,
#         show = True,
#         bins=np.linspace( vals.min(),vals.max(),10),
#
#         #choosable parameters
#         fill_color="PuBu",
#         fill_opacity=0.42,
#         line_opacity=0
#       )
#     c.add_to(m)
#
#     if show_stations:
#         L= [myProjUTM(x[0],x[1]) for x in arr_stations]
#         if len(L_station_names)>0 and len(L_station_names)==len(L):
#             L = [L[i]+[L_station_names[i]] for i in range(len(L))]
#         else:
#             L = [L[i]+[str(i)] for i in range(len(L))]
#         D = { 'own': {'list':L,
#                       'color':'blue',
#                       'layer':0}}
#         m=addDictPoints2Map(D,m)
#
#     #save file
#     m.save(name_method+".html")
#
#
#
# '''
# def getInterpolationValues(D_vals, loc, r, n_steps, arr_stations, arr_grid, s_method):
#     arr_grid = getGrid(D_vals[list(D_vals.keys())[0]], loc,r, n_steps)
#     modell = getInterpolation(arr_stations, arr_grid, s_method)
#     return {t:modell(vals) for t, vals in D_vals.items()}
#     #there is only one row in the values vals
#     #dict als eingabeparamter für vals mit zeitstempel als key
# '''
#
#

