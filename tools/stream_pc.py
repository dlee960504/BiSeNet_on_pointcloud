#from paraview.simple import *

reader = vv.getReader()
cloudInfo = reader.GetClientSideObject().GetOutput()
myintensity = cloudInfo.GetPointData().GetArray('Intensity')
query = 'intensity>50'
vv.smp.SelectPoints(query, reader)
vv.smp.Render()


project_dir = '/home/vision/project/BiSeNet/test/'
vv.saveCSVCurrentFrameSelection(project_dir + 'input_stream.csv')