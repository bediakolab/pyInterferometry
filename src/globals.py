
# material name is key
# value is list of [ lattice const in nm, PR if parallel, PR if antiparrallel, piezoelectric constant in C/m ] (PR=poisson ratio)
# Ref: HQ Graphene for lattice constants
# Ref: Zeng, Fan, Wei-Bing Zhang, and Bi-Yu Tang. Chinese Physics B 24.9 (2015): 097103 for PR
#      Assuming AP all XMMX, P all XM, ratios from DFT (optB88)
known_materials = {
	"mos2" :  [0.315, 0.234, 0.247, 2.9e-10],
	"mose2" : [0.329, 0.223, 0.230, 3.1e-10],
	"mote2" : [0.353, 0.248, 0.242, 4.3e-10],
	"wse2" :  [0.328, None,  None,  None]
}

# key is ID, value is message given to user when defining it 
data_quality_flags = {
	"good"            : "all good! - fit and/or data might still be noisy but FOV looks good", 
	"ok_crop"         : "good if cropped", 
	"bad_smallFOV"    : "FOV too small", 
	"bad_tilted"      : "fatally tilted", 
	"bad_distorted"   : "fatally distorted", 
	"ok_tilted"       : "some tilting", 
	"ok_charged"      : "some charging", 
	"ok_bubbles"      : "some annealing bubbles",
	"ok_charged_tilt" : "some charging and some tilting", 
}

# key is ID, value is message given to user when defining it 
fit_quality_flags = {
	"good"          : "fit looks good", 
	"ok"            : "fit workable but not ideal, noisy", 
	"bad"           : "fit fatally bad"
}

# key is ID, value is message given to user when defining it 
partition_quality_flags = {
	"good"          : "region partition looks good", 
	"good_crop"     : "region partition good if crop", 
	"ok"            : "region partition workable but not ideal (some outliers)", 
	"bad"           : "region partition fatally bad"
}

# key is ID, value is message given to user when defining it 
unwrap_quality_flags = {
	"good"          : "unwrap looks good", 
	"ok"            : "unwrap workable in areas", 
	"bad"           : "unwrap fatally bad"
}

# what ends up in the info.txt file. Some parameters automatically determined and updated by code.
default_parameter_filler = "{}" # changed to np.nan internally where relevant
default_parameters = {
		"Material": default_parameter_filler, 
		"Orientation": default_parameter_filler, 
		"OriginalName": default_parameter_filler, 
		"NumberDisksUsed": default_parameter_filler, 
		"LatticeConstant": default_parameter_filler, 
		"LatticeMismatch": default_parameter_filler, 
		"ProbeUsed": default_parameter_filler, 
		"PoissonRatio": default_parameter_filler, 
		"PiezoChargeConstant": default_parameter_filler, 
		"ScanShapeX": default_parameter_filler, 
		"ScanShapeY": default_parameter_filler, 
		"SmoothingSigma": default_parameter_filler, 
		"PixelSize": default_parameter_filler, 
		"FittingFunction":default_parameter_filler, 
		"RefitFromBinned":default_parameter_filler, 
		"DisplacementBasis":default_parameter_filler, 
		"UnwrapMethod": default_parameter_filler, 
		"BackgroundSubtraction": default_parameter_filler,
		"DataQualityFlag": default_parameter_filler, 
		"FitQualityFlag": default_parameter_filler, 
		"PartitionQualityFlag": default_parameter_filler, 
		"UnwrapQualityFlag": default_parameter_filler, 
		"DiffractionPatternTwist": default_parameter_filler, 
		"DiffractionPatternLatticeConst": default_parameter_filler, 
		"DiffractionPatternMismatch": default_parameter_filler, 
		"AvgMoireTwist": default_parameter_filler, 		
		"ErrMoireTwist": default_parameter_filler,     
		"AvgHeteroStrain": default_parameter_filler,    
		"ErrHeteroStrain": default_parameter_filler,   
		"AAPercent": default_parameter_filler, 
		"ABPercent": default_parameter_filler, 
		"SP1Percent": default_parameter_filler, 
		"SP2Percent": default_parameter_filler, 
		"SP3Percent": default_parameter_filler, 
		"AvgAAradius": default_parameter_filler, 
		"ErrAAradius": default_parameter_filler, 
		"AvgSPwidth": default_parameter_filler, 
		"ErrSPwidth": default_parameter_filler, 
		"AvgAAGamma": default_parameter_filler, 
		"ErrAAGamma": default_parameter_filler, 
		"AvgAAReconRot": default_parameter_filler, 
		"ErrAAReconRot": default_parameter_filler, 
		"AvgAADil": default_parameter_filler, 
		"ErrAADil": default_parameter_filler, 
		"AvgABGamma": default_parameter_filler, 
		"ErrABGamma": default_parameter_filler, 
		"AvgABReconRot": default_parameter_filler, 
		"ErrABReconRot": default_parameter_filler, 
		"AvgABDil": default_parameter_filler, 
		"ErrABDil": default_parameter_filler, 
		"AvgSPGamma": default_parameter_filler, 
		"ErrSPGamma": default_parameter_filler, 
		"AvgSPReconRot": default_parameter_filler, 
		"ErrSPReconRot": default_parameter_filler, 
		"AvgSPDil": default_parameter_filler, 
		"ErrSPDil" : default_parameter_filler
}