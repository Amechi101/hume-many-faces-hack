from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns; sns.set()
from tqdm import tqdm 
import networkx as nx
import pandas as pd 
import json
from pprint import pprint

def convertHumeJsonToFacesDataFrame(humeJsonFpath):
    """
    Converts a hume json to a faces dataframe.
    """
    with open(humeJsonFpath, "r") as f:
      data = json.loads(f.read())

    face = data[0]["results"]["predictions"][0]["models"]["face"]["grouped_predictions"]
    d_face = [(f["id"], fi["frame"], fi["box"]) for f in face if "frame" in f["predictions"][0] for fi in f["predictions"]]

    faceFrame = pd.DataFrame([f[:2] for f in d_face], columns=["Id", "Frame"])
    faceBoxes = pd.DataFrame([f[-1] for f in d_face]).rename(columns={'x':"FaceX0", 'y':"FaceY0", 'w':"FaceWidth", 'h':"FaceHeight"})
    dfFace = pd.concat([faceFrame, faceBoxes], axis=1).sort_values(by=["Frame"])
    return dfFace

def extractUniqueFaces(humeJsonFpath, totalNumberOfUniqueFaces):
    """
    Extracts 'totalNumberOfUniqueFaces' from the pool of all face ids in 
    the 'facesDataFrame'.
    Inputs:
        - facesDataFrame : pandas DataFrame
        - totalNumberOfUniqueFaces : number of unique faces to extract
    Returns:
        - Mapping between each face id and it's unique face id
    """
    ## Step 1 : make a connected graph from the similarities between all faces
    faces = convertHumeJsonToFacesDataFrame(humeJsonFpath)
    feats = faces.loc[:,["FaceWidth", "FaceHeight","FaceX0", "FaceY0", "Id"]].groupby(by=["Id"]).mean().reset_index()
    featsDict = feats.set_index("Id").T.dropna().T
    faceToFaceDistance = pd.DataFrame(cosine_similarity(featsDict, featsDict),
                                       index=featsDict.index,
                                       columns=featsDict.index)
    
    ## Get all faces that occur at the same frames
    facesByFrame = faces.groupby(by=["Frame"]).Id.apply(lambda x: list(set(x))).to_dict()
    
    ## Step 2: use the frame to frame co-occurrence to eliminate duplicates
    allPossibilities = faceToFaceDistance.copy()
    
    ## Then, remove known unique IDs:
    for frame in tqdm(facesByFrame):
        currFaces = facesByFrame[frame]
        for fi in currFaces:
            for fj in currFaces:
                if fi != fj:
                    allPossibilities.loc[fi, fj] = 0
                    
    ## Step 3 : remove lowest similarity components
    ## Start by building the networkx graph.
    G = nx.Graph()
    edges = {}
    for i in tqdm(allPossibilities.index, desc="building graph..."):
        for j in allPossibilities.columns:
            v = allPossibilities.loc[i,j]
            if v != 0:
                G.add_edge(i, j, weight=v)
                edges.update({f"{i}-{j}":v})
    edgeRanking = pd.Series(edges).sort_values()
    numFaces = len(list(nx.connected_components(G)))
    numNodes = len(nx.nodes(G))
    while numFaces < totalNumberOfUniqueFaces and numFaces < numNodes:
        ## Remove the least similar edge 
        leastSimilar = edgeRanking.index[0].split("-")
        edgeRanking = edgeRanking.drop(index=edgeRanking.index[0])
        if G.has_edge(leastSimilar[0], leastSimilar[1]):
            G.remove_edge(leastSimilar[0], leastSimilar[1])
        numFaces = len(list(nx.connected_components(G)))
    
    ## Step 4: Make a dictionary mapping of face_id's
    components = list(nx.connected_components(G))
    jsonObject = {}
    for i, c in enumerate(tqdm(components, desc="making json object...")):
        jsonObject.update({f"unique_face_{i}":c})
    return jsonObject