import os
import unittest
import vtk, qt, ctk, slicer
from qt import QSlider
from slicer.ScriptedLoadableModule import *
import logging
import slicer.util
import math
import numpy as np
import time
import sitkUtils
import SimpleITK as sitk
import SegmentStatistics
try: 
  import cv2
except ImportError:
  slicer.util.pip_install('opencv-python')
  import cv2 

#***in terminal write***
#self = slicer.mymod
#self.ms()
#print(self.ImNames)
#
# arrayX = np.load(path/file.nyc)
#  

# 
# vessel reconstruction
#

class ICEDoppler(ScriptedLoadableModule):
  """
  Basic description 
  """
  
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ICEDoppler"
    self.parent.categories = ["Conavi"]
    self.parent.dependencies = []
    self.parent.contributors = ["Hareem Nisar (VASST Lab, Western Uni)"] 
    self.parent.helpText = """help"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """ Conavi, Djalal, VASST """

#
# Widget
#

class ICEDopplerWidget(ScriptedLoadableModuleWidget):
  """
  GUI control 
  """
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    slicer.mymod = self
    self.logic = ICEDopplerLogic()
 
  def setup(self):
    # this is the function that implements all GUI
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = ICEDopplerLogic()
    self.scalarVol =  None
    self.loc = None
    
    #member variables
    
    self.imgCorr0 = np.eye(4) #to fix origin
    self.imgCorr0[0,3] = -424
    self.imgCorr0[1,3] = -424
    self.segLog = slicer.modules.segmentations.logic()
    self.MHN_CA = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelHierarchyNode')
    self.APD_CA = vtk.vtkAppendPolyData()
    self.BMF = sitk.BinaryMorphologicalClosingImageFilter()
    self.BMF.SetKernelType(sitk.sitkBall)
    self.BMF.SetKernelRadius([30,30,30])
    self.GF = sitk.SmoothingRecursiveGaussianImageFilter()
    self.GF.SetSigma([0.5, 0.5,0.5])
    self.resliceLogic = slicer.modules.volumereslicedriver.logic()
    self.WL = sitk.IntensityWindowingImageFilter() 
    self.WL.SetWindowMinimum(10)
    self.WL.SetWindowMaximum(100)
    self.transformNode = None 
    self.imageNode = None    
    self.NN = 0
    
    # Instantiate and connect widgets ...
    
    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "2.5D Reconstruction"
    self.layout.addWidget(parametersCollapsibleButton)
    
    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
    
    #
    # input Image sequence selector
    #
    self.imgSeqSelector = slicer.qMRMLNodeComboBox()
    self.imgSeqSelector.nodeTypes = ["vtkMRMLSequenceNode"]
    self.imgSeqSelector.selectNodeUponCreation = True
    self.imgSeqSelector.addEnabled = True
    self.imgSeqSelector.removeEnabled = True
    self.imgSeqSelector.noneEnabled = True
    self.imgSeqSelector.showHidden = False
    self.imgSeqSelector.showChildNodeTypes = False
    self.imgSeqSelector.setMRMLScene( slicer.mrmlScene )
    self.imgSeqSelector.setToolTip( "Select the cropped image sequence for segmentation" )
    parametersFormLayout.addRow("Image Sequence: ", self.imgSeqSelector)
    
    #
    # input Transform sequence selector
    #
    self.tfmSeqSelector = slicer.qMRMLNodeComboBox()
    self.tfmSeqSelector.nodeTypes = ["vtkMRMLSequenceNode"]
    self.tfmSeqSelector.selectNodeUponCreation = True
    self.tfmSeqSelector.addEnabled = True
    self.tfmSeqSelector.removeEnabled = True
    self.tfmSeqSelector.noneEnabled = True
    self.tfmSeqSelector.showHidden = False
    self.tfmSeqSelector.showChildNodeTypes = False
    self.tfmSeqSelector.setMRMLScene( slicer.mrmlScene )
    self.tfmSeqSelector.setToolTip( "Select the probe transform sequence" )
    parametersFormLayout.addRow("Probe Transform Sequence: ", self.tfmSeqSelector)
    
    #
    # calib Transform  selector
    #
    self.tfmSelector = slicer.qMRMLNodeComboBox()
    self.tfmSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.tfmSelector.selectNodeUponCreation = True
    self.tfmSelector.noneEnabled = False
    self.tfmSelector.showHidden = False
    self.tfmSelector.showChildNodeTypes = True
    self.tfmSelector.setMRMLScene( slicer.mrmlScene )
    self.tfmSelector.setToolTip( "Select the calibration transform" )
    parametersFormLayout.addRow("Calibration Transform: ", self.tfmSelector)
    
    #
    # GT model  selector
    #
    self.modelSelector = slicer.qMRMLNodeComboBox()
    self.modelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.modelSelector.selectNodeUponCreation = True
    self.modelSelector.noneEnabled = True
    self.modelSelector.showHidden = False
    self.modelSelector.showChildNodeTypes = False
    self.modelSelector.setMRMLScene( slicer.mrmlScene )
    self.modelSelector.setToolTip( "Select the GT model for vena contracta" )
    parametersFormLayout.addRow("Ground truth model: ", self.modelSelector)
    
    #numOfSamples
    self.numOfSamples = ctk.ctkSliderWidget()
    self.numOfSamples.decimals = 0
    self.numOfSamples.singleStep = 1
    self.numOfSamples.minimum = 0
    self.numOfSamples.maximum = 15 # numebr of frames per second 
    self.numOfSamples.value =  4
    self.numOfSamples.setToolTip("Choose numOfSamples to isolate")
    parametersFormLayout.addRow("Number of samples to isolate:",self.numOfSamples)
    
     #angle phi
    self.anglePhi = ctk.ctkSliderWidget()
    self.anglePhi.decimals = 0
    self.anglePhi.singleStep = 1
    self.anglePhi.minimum = 30
    self.anglePhi.maximum = 90 # numebr of frames per second 
    self.anglePhi.value =  75
    self.anglePhi.setToolTip("Choose anglePhi for 3D localization")
    parametersFormLayout.addRow("Imaging angle Phi:",self.anglePhi)
    
    
    #
    # check box to enable auto-update
    #
    self.autoUpdateCheckBox = qt.QCheckBox()
    self.autoUpdateCheckBox.checked = 0
    self.autoUpdateCheckBox.setToolTip("If checked, auto-update the output volume when inout is changed")
    #parametersFormLayout.addRow("Auto-update Output Volume", self.autoUpdateCheckBox)
    
    
    #
    # Apply Doppler Button
    #
    self.getSystolicDopplerButton = qt.QPushButton("Get Systolic Doppler only")
    self.getSystolicDopplerButton.toolTip = "colored parts are separated from grays"
    self.getSystolicDopplerButton.enabled = True
    parametersFormLayout.addRow(self.getSystolicDopplerButton)
    
    #
    # Add Doppler Button
    #
    self.addDopplerButton = qt.QPushButton("Add Systolic Doppler Sequence")
    self.addDopplerButton.toolTip = "keep the max of all frames."
    self.addDopplerButton.enabled = True
    parametersFormLayout.addRow(self.addDopplerButton)
    
    
    #
    # Convert vector to scalar Button
    #
    self.convertToScalarButton = qt.QPushButton("Convert to Scalar")
    self.convertToScalarButton.toolTip = "covert the -added doppler volume- to scalar via luminance method"
    self.convertToScalarButton.enabled = True
    parametersFormLayout.addRow(self.convertToScalarButton)
    
    #
    # Get the jet via thresholding and keeping the island
    #
    self.getJetButton = qt.QPushButton("Get the regurg. jet")
    self.getJetButton.toolTip = "Get the jet via thresholding and keeping the island"
    self.getJetButton.enabled = True
    parametersFormLayout.addRow(self.getJetButton)
    
    #
    # Get the 3D location from 2D image fiducial
    #
    self.get3DlocationButton = qt.QPushButton("3D localization")
    self.get3DlocationButton.toolTip = "Get the 3D location from 2D image fiducial"
    self.get3DlocationButton.enabled = True
    parametersFormLayout.addRow(self.get3DlocationButton)
    
    #
    # Get the 3D location from 2D image fiducial
    #
    self.getErrorButton = qt.QPushButton("Error from GT")
    self.getErrorButton.toolTip = "Get the distance of fiducial from the GT model"
    self.getErrorButton.enabled = True
    parametersFormLayout.addRow(self.getErrorButton)
    
    # connections
    
    #self.imgSeqSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onImageChanged)
    #self.tfmSeqSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onTransformChanged)
    #self.tfmSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onCalibrationSelected)
    
    self.getSystolicDopplerButton.connect('clicked(bool)', self.onGetSystolicDopplerButton)
    self.addDopplerButton.connect('clicked(bool)', self.onAddDopplerButton)
    self.convertToScalarButton.connect('clicked(bool)', self.onConvertToScalarButton)
    self.getJetButton.connect('clicked(bool)', self.onGetJetButton)
    self.get3DlocationButton.connect('clicked(bool)', self.onGet3DlocationButton)
    self.getErrorButton.connect('clicked(bool)', self.onGetErrorButton)
    
    # Add vertical spacer
    self.layout.addStretch(1)
    
    
    # Refresh Apply button state
    #self.onSelect()
  def onImageChanged(self):
    self.imageNode = self.imgSeqSelector.currentNode() 
    if self.imageNode is None:
      print('Please select an Image Sequence')
    else:
      self.seq = self.imgSeqSelector.currentNode()
      #self.numOfSamples.maximum = 15 # number of frames per second 
      
  def onTransformChanged(self):
    self.transformNode = self.tfmSeqSelector.currentNode() 
    if self.transformNode is None:
      print('Please select a Probe Sequence')
    
  def onGetSystolicDopplerButton(self):
    self.logic.getSystolicDopplerSequence(self.imgSeqSelector.currentNode(), self.tfmSeqSelector.currentNode(), np.int(self.numOfSamples.value))
  
  def onAddDopplerButton(self):
    self.logic.addDopplerSequence()
    
  def onConvertToScalarButton(self):
    self.logic.convertToScalar()
    
  def onGetJetButton(self):
    self.logic.segmentEditorEffects()
  
  def onGet3DlocationButton(self):
    #catching the existing sequence browser to find the current index
    #self.inputVolNode  = self.imgSeq.GetNthDataNode(0)
    #self.seqBrwsr = slicer.util.getNode(self.inputVolNode.GetName()[:-6]) ## if you know the image volume. 
    AllSeqBrwsr = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
    seqBrwsr = AllSeqBrwsr[0]
    itemNum = seqBrwsr.GetSelectedItemNumber()
    
    tfmSeqNode = self.tfmSeqSelector.currentNode()
    tfmNode = tfmSeqNode.GetNthDataNode(itemNum)
    
    self.loc = self.logic.find3DVC(tfmNode, self.tfmSelector.currentNode(), np.int(self.anglePhi.value))
  
  def onGetErrorButton(self):
    modelNode = self.modelSelector.currentNode()
    markupsNode = self.loc
    
    if modelNode and markupsNode is not None:
      error = self.logic.findError(modelNode, markupsNode)
      print("Distance from GT (in mm): ", error)
    else:
      print("Please select a ground truth model first")
    
    return 0
    
  def cleanup(self):
    print("in clean up")
#
#  Logic
#

class ICEDopplerLogic(ScriptedLoadableModuleLogic):
  """Functions come here
  """
  
  def __init__(self):
    self.volumesLogic = slicer.modules.volumes.logic()
  
  def findError(self, modelNode, markupsNode):
    # Transform model polydata to world coordinate system
    if modelNode.GetParentTransformNode():
      transformModelToWorld = vtk.vtkGeneralTransform()
      slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(modelNode.GetParentTransformNode(), None, transformModelToWorld)
      polyTransformToWorld = vtk.vtkTransformPolyDataFilter()
      polyTransformToWorld.SetTransform(transformModelToWorld)
      polyTransformToWorld.SetInputData(modelNode.GetPolyData())
      polyTransformToWorld.Update()
      surface_World = polyTransformToWorld.GetOutput()
    else:
      surface_World = modelNode.GetPolyData()
    
    # Create arrays to store results
    indexCol = vtk.vtkIntArray()
    indexCol.SetName("Index")
    labelCol = vtk.vtkStringArray()
    labelCol.SetName("Name")
    distanceCol = vtk.vtkDoubleArray()
    distanceCol.SetName("Distance")

    distanceFilter = vtk.vtkImplicitPolyDataDistance()
    distanceFilter.SetInput(surface_World);
    nOfFiduciallPoints = markupsNode.GetNumberOfFiducials()
    for i in range(0, nOfFiduciallPoints):
      point_World = [0,0,0]
      markupsNode.GetNthControlPointPositionWorld(i, point_World)
      closestPointOnSurface_World = [0,0,0]
      closestPointDistance = distanceFilter.EvaluateFunctionAndGetClosestPoint(point_World, closestPointOnSurface_World)
      indexCol.InsertNextValue(i)
      labelCol.InsertNextValue(markupsNode.GetNthControlPointLabel(i))
      distanceCol.InsertNextValue(closestPointDistance)
    
    # Create a table from result arrays
    resultTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Points from surface distance")
    resultTableNode.AddColumn(indexCol)
    resultTableNode.AddColumn(labelCol)
    resultTableNode.AddColumn(distanceCol)
    
    # Show table in view layout
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpTableView)
    slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveTableID(resultTableNode.GetID())
    slicer.app.applicationLogic().PropagateTableSelection()
    
    return closestPointDistance
    
  def find3DVC(self, probeTfmNode, calibTfm, anglePhi):
    
    loc2D = slicer.util.getNode("VC-2D")
    fid2D = [0,0,0]
    loc2D.GetNthFiducialPosition(0, fid2D)
    
    print("in 2D: ",fid2D)
    
    x = fid2D[0] - 424
    y = fid2D[1] - 424
    d = np.sqrt((x*x) + (y*y)) # 6.6 us for X= np.zeros(200) 
    z = d*math.tan(math.radians(90-anglePhi))
    
    loc3D = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "VC-3D")
    loc3D.AddFiducial(x, y, z)
    
    print("in 3D", x,y,z)
    
    #ApplyTfmsToFiducial(self,  phi):
    
    #loc3D = slicer.util.getNode("VC-3D")
    fid3D = [0,0,0]
    loc3D.GetNthFiducialPosition(0, fid3D)
    fd = loc3D.GetDisplayNode()
    fd.SetGlyphScale(5)
    fid3D = np.append(fid3D, np.ones(1))
    
    #probeTfmNode = probeTfmSeq.GetNthDataNode(i)
    probeArr = slicer.util.arrayFromTransformMatrix(probeTfmNode)
    calibArr = slicer.util.arrayFromTransformMatrix(calibTfm)  
    #apply, imgCorr, then calibration matrix, and then probe position transform
    ##OutputTfm = np.matmul(Parent, Child)
    #calibCorr = np.matmul(calibArr, imgCorr)
    tfmArr = np.matmul(probeArr, calibArr)  
    fid3DArr = np.matmul(tfmArr, fid3D)[0:3] #ignore the last 1
    
    #make a new fiducial node
    locVC = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "VC-EM")
    locVC.AddFiducialFromArray(fid3DArr)
    locVC.SetLocked(1)
    fd = locVC.GetDisplayNode()
    fd.SetGlyphScale(5)
    fd.SetSelectedColor([0,1,0])
    fd.SetTextScale(3.7)
    print("in EM ",fid3DArr)
    
    return locVC
  
  def findVC(self, lm, l0, l1):
    
    arr = slicer.util.arrayFromVolume(lm)
    arr = np.squeeze(arr)
    
    #print(l0[0], l0[1])
    #print(l1[0], l1[1])
    
    if l0[0]>0 and l0[1]<0:
      print("flip")
      l0[0] = -1*l0[0]
      l0[1] = -1*l0[1]
      l1[0] = -1*l1[0]
      l1[1] = -1*l1[1]
    elif l0[0]>0 and l0[1]>0 and l1[0]<0 and l1[1]>0:
      print("flipping it all")
      l0[0] = -1*l0[0]
      l0[1] = -1*l0[1]
      l1[0] = -1*l1[0]
      l1[1] = -1*l1[1]
    
    [x,y]  = np.where(arr>0)
    xn = x-np.mean(x)
    yn = y-np.mean(y)
    xyn = np.stack((xn, yn), axis=-1) ## shape = nx2
    
    lambda0 = np.array((l0[0], l0[1]))#axis with minor principal moments
    lambda1 = np.array((l1[0], l1[1]))  #axis with major principal moments 
    lambdas = np.stack((lambda0, lambda1), axis=0) # stack on top-bottom
    
    
    XYN= np.matmul(xyn,lambdas) # shape = nx2
    XN = XYN[:,0]
    YN = XYN[:,1]
    
    XR = np.round(XN)
    size = np.max(XR)-np.min(XR)+1
    XYmm = np.zeros([np.int(size),4])
    minXR = np.int(min(XR))
    maxXR = np.int(max(XR))
    ## taverse each possible value of XR
    for i in range(minXR,maxXR+1):
      index = i-minXR
      
      #go through the entire XR, where XR==i, 
      #keep the corresponding YN values in array 
        
      XRpos = np.where(XR==i) #gives indexes to look for in YN
      Ytally = YN[XRpos]      #stores all corresponding values. 
      
      #now finding min and max to find the distance and location
      # XYmm is the output that stores the value of X, min Y, max Y, 
      #and corresponding distance betweeen min/max  
      
      dist = np.min(np.matmul(Ytally,(Ytally>0)))-np.max(np.matmul(Ytally,(Ytally<0)))
      XYmm[index,:] = np.array([i, min(Ytally), max(Ytally),np.max(Ytally)-np.min(Ytally)])
      
    #end for 
    
    p = np.where(XYmm[:,3]>10) #make this 30
    pArr = np.asarray(p)
    result_index = pArr[0,0]
    
    result = XYmm[result_index,:]
    result =np.squeeze(result)
    end1 = np.array([result[0], result[1]])
    end2 = np.array([result[0], result[2]])
    vcw = result[3]
    
    #converting the two end points back to original coordinates 
    XYends = np.stack((end1, end2), axis=0) # stack on top-bottom
    xyends = np.matmul(XYends,np.transpose(lambdas))
    xyends = np.stack((xyends[:,0]+np.mean(x), xyends[:,1]+np.mean(y)), axis=-1)
    xymid  = np.mean(xyends, axis=0)
    
    print(xymid)
    
    loc2D = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "VC-2D")
    loc2D.AddFiducial(xymid[1],xymid[0],0)
    fd = loc2D.GetDisplayNode()
    fd.SetGlyphScale(5)
    fd.SetSelectedColor([1,0,0])
    fd.SetTextScale(3.7)
    #slicer.modules.markups.logic().AddFiducial(xymid[1],xymid[0],0)
  
  
  def getSegmentStatistics(self, segmentationNode):
    segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
    segStatLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.elongation.enabled", str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.principal_moments.enabled", str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.principal_axis_x.enabled", str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.principal_axis_y.enabled", str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.principal_axis_z.enabled", str(True))
    segStatLogic.computeStatistics()
    stats = segStatLogic.getStatistics()
    
    for segmentId in stats["SegmentIDs"]:
      elongation = stats[segmentId,"LabelmapSegmentStatisticsPlugin.elongation"]
      pm = stats[segmentId,"LabelmapSegmentStatisticsPlugin.principal_moments"]
      pax = stats[segmentId,"LabelmapSegmentStatisticsPlugin.principal_axis_x"]
      self.pay = stats[segmentId,"LabelmapSegmentStatisticsPlugin.principal_axis_y"]
      self.paz = stats[segmentId,"LabelmapSegmentStatisticsPlugin.principal_axis_z"]    
    
    print("segmentat statistics! - PCA")
    print("principal moments", pm)
    print("pay", self.pay)
    print("paz", self.paz)
    
    self.labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "lm")
    #segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName("jet")
    slicer.vtkSlicerSegmentationsModuleLogic.ExportVisibleSegmentsToLabelmapNode(segmentationNode, self.labelmapVolumeNode, slicer.util.getNode("Scalar Doppler Volume"))
    
    #finding the vena contracta 
    #self.findVC(self.labelmapVolumeNode, self.pay, self.paz)
    
  
  def segmentEditorEffects(self):
    # Create segmentation
    
    masterNode = slicer.util.getNode("Scalar Doppler Volume")
    
    segmentationNode = slicer.vtkMRMLSegmentationNode()
    slicer.mrmlScene.AddNode(segmentationNode)
    segmentationNode.CreateDefaultDisplayNodes() # only needed for display
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterNode)
    addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment("jet")
    # Create segment editor to get access to effects
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setMasterVolumeNode(masterNode)
    
    # Thresholding
    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MinimumThreshold",150)
    #effect.setParameter("MaximumThreshold",250)
    effect.self().onApply()
    
    self.getSegmentStatistics(segmentationNode)
    
    # Smoothing
    segmentEditorWidget.setActiveEffectByName("Smoothing")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("SmoothingMethod", "CLOSING")
    effect.setParameter("KernelSizeMm", 10)
    ##effect.self().onApply()
    
    # Find largest object, remove all other regions from the segment
    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MinimumSize", 30)
    effect.setParameter("Operation", "KEEP_LARGEST_ISLAND")
    effect.self().onApply()
    
    
    # segmentEditorWidget.setActiveEffectByName("Logical operators")
    # effect = segmentEditorWidget.activeEffect()
    # effect.setParameter("Operation",'UNION')
    # modifierSegmentID = segNode.GetSegmentation().GetSegmentIdBySegmentName("bone")
    # effect.setParameter("ModifierSegmentID", modifierSegmentID)
    # effect.self().onApply()
    
    print("regurgitant jet!")
    
    
    
    #finding the vena contracta 
    slicer.vtkSlicerSegmentationsModuleLogic.ExportVisibleSegmentsToLabelmapNode(segmentationNode, self.labelmapVolumeNode, slicer.util.getNode("Scalar Doppler Volume"))
    
    self.findVC(self.labelmapVolumeNode, self.pay, self.paz)
    
  def convertToScalar(self):
    inputVolumeNode = slicer.util.getNode("added Doppler volume")
    outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', "Scalar Doppler Volume")
    
    ijkToRAS = vtk.vtkMatrix4x4()
    inputVolumeNode.GetIJKToRASMatrix(ijkToRAS)
    outputVolumeNode.SetIJKToRASMatrix(ijkToRAS)

    extract = vtk.vtkImageExtractComponents()
    extract.SetInputConnection(inputVolumeNode.GetImageDataConnection())
    extract.SetComponents(0, 1, 2)
    luminance = vtk.vtkImageLuminance()
    luminance.SetInputConnection(extract.GetOutputPort())
    luminance.Update()
    outputVolumeNode.SetImageDataConnection(luminance.GetOutputPort())
    
    red_logic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
    red_cn = red_logic.GetSliceCompositeNode()
    red_logic.GetSliceCompositeNode().SetBackgroundVolumeID(outputVolumeNode.GetID())
    
    print("scalar conversion done")
    
    return outputVolumeNode
  
  def addDopplerSequence(self):
    #generating a new doppler volume
    self.inputVolNode  = self.dopplerSeqNode.GetNthDataNode(0)
    self.dopplerVol = self.volumesLogic.CloneVolume(slicer.mrmlScene, self.inputVolNode, "added Doppler volume")
    self.dopplerArr = slicer.util.arrayFromVolume(self.dopplerVol) ##(1, 849, 849, 3)
    
    self.mx  = self.dopplerSeqNode.GetNumberOfDataNodes()
    self.sumArr = np.zeros([self.dopplerArr.shape[1],self.dopplerArr.shape[2],3,self.mx]) ##(849,849,3,max)
    
    for i in range(0, self.mx):
      #getting input volume for segmentation
      self.inputVolNode  = self.dopplerSeqNode.GetNthDataNode(i)
      self.seqIndex      = self.dopplerSeqNode.GetNthIndexValue(i)
      
      #add consecutively.
      self.inputVolNode = self.getRedVol(self.inputVolNode)
      arr = slicer.util.arrayFromVolume(self.inputVolNode)
      self.sumArr[:,:,:,i] = arr #(849, 849, 3, max) in the end
      
    #end for
    
    self.maxArr = np.max(self.sumArr, 3)#(849, 849, 3)
    self.dopplerArr[0,:,:,:] = self.maxArr #(1, 849, 849, 3)
    slicer.util.updateVolumeFromArray(self.dopplerVol, self.dopplerArr)
    #self.dopplerSeqNode.SetDataNodeAtValue(  self.dopplerVol, self.seqIndex+1) 
    
    print('done')
    
  def getSystolicDopplerSequence(self, img, tfm, numOfSamples):
    
    
    #create a new sequence browser and image sequence
    self.dopplerSBNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', 'Systolic Doppler SeqBrwsr')
    self.dopplerSBNode.SetScene(slicer.mrmlScene)
    slicer.mrmlScene.AddNode(self.dopplerSBNode)
    
    
    # creating sequences to store data processes
    self.dopplerSeqNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', 'Doppler Image Sequence')
    self.probeSeqNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', 'Doppler ProbeTfm Sequence')
        
    #getting the original sequences to copy from
    self.imgSeq = img
    print(self.imgSeq)
    self.tfmSeq = tfm
    
    #catching the existing sequence browser to find the current index
    #self.inputVolNode  = self.imgSeq.GetNthDataNode(0)
    #self.seqBrwsr = slicer.util.getNode(self.inputVolNode.GetName()[:-6]) ## if you know the image volume. 
    self.AllSeqBrwsr = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
    self.seqBrwsr = self.AllSeqBrwsr[0]
    self.itemNum = self.seqBrwsr.GetSelectedItemNumber()
    #currentSeqIndex  = self.imgSeq.GetNthIndexValue(self.itemNum)
    
    
    startingIndex = self.itemNum
    maxIndex = 0
    if self.imgSeq is not None:
      maxIndex =  self.imgSeq.GetNumberOfDataNodes() 
    gap = 15 #pick up a sample every 15th image
    sp = numOfSamples
    for k in range(startingIndex, maxIndex, gap):
      for j in range (0, sp):
        i = j+k
        print(i)
        if i< maxIndex:
          #getting input volume for segmentation
          self.inputVolNode  = self.imgSeq.GetNthDataNode(i)
          self.inputTfmNode  = self.tfmSeq.GetNthDataNode(i)
          self.seqIndex      = self.imgSeq.GetNthIndexValue(i)
          #print(self.inputVolNode.GetName())
          
          self.tempVol = self.volumesLogic.CloneVolume(slicer.mrmlScene, self.inputVolNode, "Temp volume")
          self.tempVol = self.getDopplerVol(self.tempVol)
          
          #Reconstructing 3D LabelMap form
          #self.logic.reconstruction_VOL_Vector(self.tempVol, self.tempVol, 73) #original line messing up orientation
          
          #Push volume to  sequences for storage and viewing 
          self.dopplerSeqNode.SetDataNodeAtValue(self.tempVol, self.seqIndex) 
          self.probeSeqNode.SetDataNodeAtValue(self.inputTfmNode, self.seqIndex)
          
          slicer.mrmlScene.RemoveNode(self.tempVol)
      
    #end for
    
    #adding the selected image from sequences to the 'doppler sequence browser' to view separately
    self.dopplerSBNode.AddProxyNode(self.dopplerSeqNode.GetNthDataNode(0),  self.dopplerSeqNode)
    self.dopplerSBNode.AddProxyNode(self.probeSeqNode.GetNthDataNode(0),  self.probeSeqNode)
  
  def getDopplerVol(self, volnode):
    a = slicer.util.arrayFromVolume(volnode)
    b = np.squeeze(a)
    R = b[:,:,0]
    G = b[:,:,1]
    B = b[:,:,2]

    a1 = (R==G)
    a2 = (G==B)
    graysLocation = np.where(a1&a2)

    b[graysLocation] = 0  ## this will automatically update the imaging volume 
    
    return volnode
    
  def getRedVol(self, volnode):
    a = slicer.util.arrayFromVolume(volnode)
    b = np.squeeze(a)
    R = b[:,:,0]
    G = b[:,:,1]
    B = b[:,:,2]
    
    blueLocation = np.where(B)

    b[blueLocation] = 0  ## this will automatically update the imaging volume 
    
    return volnode
  