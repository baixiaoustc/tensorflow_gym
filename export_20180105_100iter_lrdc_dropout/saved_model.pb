Ňć
 ĺ
9
Add
x"T
y"T
z"T"
Ttype:
2	
ë
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
Č
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
î
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
í
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Square
x"T
y"T"
Ttype:
	2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.2.12v1.2.0-1740-gafa03e2´
P
PlaceholderPlaceholder*
_output_shapes
:*
shape:*
dtype0
R
Placeholder_1Placeholder*
_output_shapes
:*
shape:*
dtype0
W
bias/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *   @
h
bias
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 

bias/AssignAssignbiasbias/initial_value*
use_locking(*
T0*
_class
	loc:@bias*
validate_shape(*
_output_shapes
: 
U
	bias/readIdentitybias*
T0*
_class
	loc:@bias*
_output_shapes
: 
l
random_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
W
random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
Y
random_normal/stddevConst*
valueB
 *
×#<*
_output_shapes
: *
dtype0
Ś
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*

seed *
T0*&
_output_shapes
:*
seed2 

random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*&
_output_shapes
:*
T0
l
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*&
_output_shapes
:

w_conv1
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
Ś
w_conv1/AssignAssignw_conv1random_normal*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:*
_class
loc:@w_conv1
n
w_conv1/readIdentityw_conv1*
_class
loc:@w_conv1*&
_output_shapes
:*
T0
n
random_normal_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
Y
random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
×#<
Ş
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*

seed *
T0*
dtype0*&
_output_shapes
:*
seed2 

random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*&
_output_shapes
:*
T0
r
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*&
_output_shapes
:

b_conv1
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
¨
b_conv1/AssignAssignb_conv1random_normal_1*
_class
loc:@b_conv1*&
_output_shapes
:*
T0*
validate_shape(*
use_locking(
n
b_conv1/readIdentityb_conv1*
T0*&
_output_shapes
:*
_class
loc:@b_conv1
n
random_normal_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             
Y
random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_2/stddevConst*
valueB
 *
×#<*
dtype0*
_output_shapes
: 
Ş
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*&
_output_shapes
: *
seed2 *
dtype0*
T0*

seed 

random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0*&
_output_shapes
: 
r
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0*&
_output_shapes
: 

w_conv2
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
¨
w_conv2/AssignAssignw_conv2random_normal_2*
use_locking(*
T0*
_class
loc:@w_conv2*
validate_shape(*&
_output_shapes
: 
n
w_conv2/readIdentityw_conv2*
T0*&
_output_shapes
: *
_class
loc:@w_conv2
n
random_normal_3/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             
Y
random_normal_3/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
[
random_normal_3/stddevConst*
valueB
 *
×#<*
dtype0*
_output_shapes
: 
Ş
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*

seed *
T0*
dtype0*&
_output_shapes
: *
seed2 

random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
T0*&
_output_shapes
: 
r
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0*&
_output_shapes
: 

b_conv2
VariableV2*&
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
¨
b_conv2/AssignAssignb_conv2random_normal_3*&
_output_shapes
: *
validate_shape(*
_class
loc:@b_conv2*
T0*
use_locking(
n
b_conv2/readIdentityb_conv2*
T0*
_class
loc:@b_conv2*&
_output_shapes
: 
f
random_normal_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Y
random_normal_4/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
[
random_normal_4/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
¤
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
dtype0*

seed *
T0* 
_output_shapes
:
 *
seed2 

random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev* 
_output_shapes
:
 *
T0
l
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean* 
_output_shapes
:
 *
T0

	w_affine1
VariableV2*
shared_name *
dtype0*
shape:
 * 
_output_shapes
:
 *
	container 
¨
w_affine1/AssignAssign	w_affine1random_normal_4* 
_output_shapes
:
 *
validate_shape(*
_class
loc:@w_affine1*
T0*
use_locking(
n
w_affine1/readIdentity	w_affine1*
T0*
_class
loc:@w_affine1* 
_output_shapes
:
 
`
random_normal_5/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_5/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
[
random_normal_5/stddevConst*
valueB
 *
×#<*
_output_shapes
: *
dtype0

$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
_output_shapes	
:*
seed2 *
T0*

seed *
dtype0
~
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
_output_shapes	
:*
T0
g
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
T0*
_output_shapes	
:
w
	b_affine1
VariableV2*
shared_name *
dtype0*
shape:*
_output_shapes	
:*
	container 
Ł
b_affine1/AssignAssign	b_affine1random_normal_5*
use_locking(*
T0*
_class
loc:@b_affine1*
validate_shape(*
_output_shapes	
:
i
b_affine1/readIdentity	b_affine1*
_output_shapes	
:*
_class
loc:@b_affine1*
T0
f
random_normal_6/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
Y
random_normal_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_6/stddevConst*
valueB
 *
×#<*
dtype0*
_output_shapes
: 
Ł
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
dtype0*

seed *
T0*
_output_shapes
:	*
seed2 

random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0*
_output_shapes
:	
k
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
_output_shapes
:	*
T0

	w_affine2
VariableV2*
shared_name *
dtype0*
shape:	*
_output_shapes
:	*
	container 
§
w_affine2/AssignAssign	w_affine2random_normal_6*
_output_shapes
:	*
validate_shape(*
_class
loc:@w_affine2*
T0*
use_locking(
m
w_affine2/readIdentity	w_affine2*
_output_shapes
:	*
_class
loc:@w_affine2*
T0
_
random_normal_7/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_7/stddevConst*
valueB
 *
×#<*
dtype0*
_output_shapes
: 

$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
_output_shapes
:*
seed2 *
dtype0*
T0*

seed 
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
T0*
_output_shapes
:
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
T0*
_output_shapes
:
u
	b_affine2
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
˘
b_affine2/AssignAssign	b_affine2random_normal_7*
_class
loc:@b_affine2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
h
b_affine2/readIdentity	b_affine2*
T0*
_class
loc:@b_affine2*
_output_shapes
:

Placeholder_2Placeholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙TT*$
shape:˙˙˙˙˙˙˙˙˙TT*
dtype0
ž
Conv2DConv2DPlaceholder_2w_conv1/read*
paddingSAME*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
addAddConv2Db_conv1/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
K
ReluReluadd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Conv2D_1Conv2DReluw_conv2/read*
paddingSAME*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
^
add_1AddConv2D_1b_conv2/read*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
O
Relu_1Reluadd_1*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
^
Reshape/shapeConst*
valueB"˙˙˙˙   *
_output_shapes
:*
dtype0
j
ReshapeReshapeRelu_1Reshape/shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

MatMulMatMulReshapew_affine1/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
W
add_2AddMatMulb_affine1/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
H
Relu_2Reluadd_2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
I
Relu_3ReluRelu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
S
dropout/ShapeShapeRelu_3*
T0*
out_type0*
_output_shapes
:
_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
_
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*

seed *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
\
dropout/addAddPlaceholder_1dropout/random_uniform*
T0*
_output_shapes
:
F
dropout/FloorFloordropout/add*
_output_shapes
:*
T0
P
dropout/divRealDivRelu_3Placeholder_1*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

MatMul_1MatMuldropout/mulw_affine2/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
X
add_3AddMatMul_1b_affine2/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
Placeholder_3Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
MulMuladd_3Placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
q
SumSumMulSum/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
	keep_dims( *

Tidx0
h
Placeholder_4Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
subSubPlaceholder_4Sum*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
C
SquareSquaresub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Y
MeanMeanSquareConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
_
gradients/Mean_grad/ShapeShapeSquare*
_output_shapes
:*
out_type0*
T0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
a
gradients/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
~
gradients/Square_grad/mul/xConst^gradients/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
p
gradients/Square_grad/mulMulgradients/Square_grad/mul/xsub*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Square_grad/mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
gradients/sub_grad/ShapeShapePlaceholder_4*
_output_shapes
:*
out_type0*
T0
]
gradients/sub_grad/Shape_1ShapeSum*
T0*
out_type0*
_output_shapes
:
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
¤
gradients/sub_grad/SumSumgradients/Square_grad/mul_1(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
gradients/sub_grad/Sum_1Sumgradients/Square_grad/mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ö
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_class#
!loc:@gradients/sub_grad/Reshape
Ü
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
[
gradients/Sum_grad/ShapeShapeMul*
_output_shapes
:*
out_type0*
T0
Y
gradients/Sum_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
n
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*
_output_shapes
: 
t
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*
_output_shapes
: 
]
gradients/Sum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
`
gradients/Sum_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
`
gradients/Sum_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
˘
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*
_output_shapes
:
_
gradients/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :
{
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*
_output_shapes
: 
Í
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
gradients/Sum_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0

gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
_output_shapes
:*
T0
§
gradients/Sum_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_1 gradients/Sum_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
T0

gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
]
gradients/Mul_grad/ShapeShapeadd_3*
T0*
_output_shapes
:*
out_type0
g
gradients/Mul_grad/Shape_1ShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
´
(gradients/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_grad/Shapegradients/Mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
w
gradients/Mul_grad/mulMulgradients/Sum_grad/TilePlaceholder_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Mul_grad/SumSumgradients/Mul_grad/mul(gradients/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/Mul_grad/ReshapeReshapegradients/Mul_grad/Sumgradients/Mul_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
q
gradients/Mul_grad/mul_1Muladd_3gradients/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
gradients/Mul_grad/Sum_1Sumgradients/Mul_grad/mul_1*gradients/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/Mul_grad/Reshape_1Reshapegradients/Mul_grad/Sum_1gradients/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/Mul_grad/tuple/group_depsNoOp^gradients/Mul_grad/Reshape^gradients/Mul_grad/Reshape_1
Ú
+gradients/Mul_grad/tuple/control_dependencyIdentitygradients/Mul_grad/Reshape$^gradients/Mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/Mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ŕ
-gradients/Mul_grad/tuple/control_dependency_1Identitygradients/Mul_grad/Reshape_1$^gradients/Mul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@gradients/Mul_grad/Reshape_1*
T0
b
gradients/add_3_grad/ShapeShapeMatMul_1*
_output_shapes
:*
out_type0*
T0
f
gradients/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients/add_3_grad/SumSum+gradients/Mul_grad/tuple/control_dependency*gradients/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ź
gradients/add_3_grad/Sum_1Sum+gradients/Mul_grad/tuple/control_dependency,gradients/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
â
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
T0
Ű
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*
_output_shapes
:*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
Ŕ
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyw_affine2/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ś
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_3_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	*
transpose_a(*
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
í
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes
:	*
T0
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0*
T0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0*
T0
Ě
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/dropout/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
_output_shapes
:*
T0
ˇ
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
_output_shapes
:*
Tshape0

 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
˝
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ś
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
_output_shapes
:*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
ë
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
_output_shapes
:
ń
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
f
 gradients/dropout/div_grad/ShapeShapeRelu_3*
T0*
_output_shapes
:*
out_type0
x
"gradients/dropout/div_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependencyPlaceholder_1*
T0*
_output_shapes
:
ť
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
°
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
`
gradients/dropout/div_grad/NegNegRelu_3*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/NegPlaceholder_1*
_output_shapes
:*
T0

$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1Placeholder_1*
T0*
_output_shapes
:
Ł
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
ť
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ś
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ű
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0
ń
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0

gradients/Relu_3_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_3*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Relu_2_grad/ReluGradReluGradgradients/Relu_3_grad/ReluGradRelu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients/add_2_grad/ShapeShapeMatMul*
_output_shapes
:*
out_type0*
T0
g
gradients/add_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
ş
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
Tshape0*
_output_shapes	
:*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
ă
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@gradients/add_2_grad/Reshape
Ü
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
_output_shapes	
:*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0
ž
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyw_affine1/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
transpose_a( *
T0
ą
gradients/MatMul_grad/MatMul_1MatMulReshape-gradients/add_2_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
 *
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ĺ
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ */
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
ă
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
 *1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
b
gradients/Reshape_grad/ShapeShapeRelu_1*
out_type0*
_output_shapes
:*
T0
ż
gradients/Reshape_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

gradients/Relu_1_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
b
gradients/add_1_grad/ShapeShapeConv2D_1*
T0*
out_type0*
_output_shapes
:
u
gradients/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"             
ş
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ľ
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
Ż
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
˘
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
: 
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ę
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ç
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*&
_output_shapes
: *1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
a
gradients/Conv2D_1_grad/ShapeShapeRelu*
T0*
out_type0*
_output_shapes
:
Ę
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/Shapew_conv2/read-gradients/add_1_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME
x
gradients/Conv2D_1_grad/Shape_1Const*%
valueB"             *
_output_shapes
:*
dtype0
˘
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterRelugradients/Conv2D_1_grad/Shape_1-gradients/add_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
: *?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0

gradients/Relu_grad/ReluGradReluGrad0gradients/Conv2D_1_grad/tuple/control_dependencyRelu*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
gradients/add_grad/ShapeShapeConv2D*
_output_shapes
:*
out_type0*
T0
s
gradients/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*&
_output_shapes
:*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
â
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_class#
!loc:@gradients/add_grad/Reshape
ß
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*&
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1
h
gradients/Conv2D_grad/ShapeShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
Ä
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/Shapew_conv1/read+gradients/add_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
v
gradients/Conv2D_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            
Ľ
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder_2gradients/Conv2D_grad/Shape_1+gradients/add_grad/tuple/control_dependency*
data_formatNHWC*
strides
*&
_output_shapes
:*
paddingSAME*
T0*
use_cudnn_on_gpu(

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙TT

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
|
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@b_affine1*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
_output_shapes
: *
dtype0*
shape: *
	container *
_class
loc:@b_affine1*
shared_name 
Ź
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@b_affine1*
validate_shape(*
_output_shapes
: 
h
beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *
_class
loc:@b_affine1
|
beta2_power/initial_valueConst*
valueB
 *wž?*
_class
loc:@b_affine1*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
	container *
dtype0*
_class
loc:@b_affine1*
_output_shapes
: *
shape: *
shared_name 
Ź
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@b_affine1
h
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
_class
loc:@b_affine1*
T0

w_conv1/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
:*
_class
loc:@w_conv1*%
valueB*    
Ź
w_conv1/Adam
VariableV2*
	container *
dtype0*
_class
loc:@w_conv1*&
_output_shapes
:*
shape:*
shared_name 
Á
w_conv1/Adam/AssignAssignw_conv1/Adamw_conv1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@w_conv1*
validate_shape(*&
_output_shapes
:
x
w_conv1/Adam/readIdentityw_conv1/Adam*
T0*&
_output_shapes
:*
_class
loc:@w_conv1
Ą
 w_conv1/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
:*
_class
loc:@w_conv1*%
valueB*    
Ž
w_conv1/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:*&
_output_shapes
:*
_class
loc:@w_conv1
Ç
w_conv1/Adam_1/AssignAssignw_conv1/Adam_1 w_conv1/Adam_1/Initializer/zeros*
_class
loc:@w_conv1*&
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
w_conv1/Adam_1/readIdentityw_conv1/Adam_1*
T0*&
_output_shapes
:*
_class
loc:@w_conv1

b_conv1/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
:*
_class
loc:@b_conv1*%
valueB*    
Ź
b_conv1/Adam
VariableV2*
	container *
dtype0*
_class
loc:@b_conv1*
shared_name *&
_output_shapes
:*
shape:
Á
b_conv1/Adam/AssignAssignb_conv1/Adamb_conv1/Adam/Initializer/zeros*
_class
loc:@b_conv1*&
_output_shapes
:*
T0*
validate_shape(*
use_locking(
x
b_conv1/Adam/readIdentityb_conv1/Adam*
T0*
_class
loc:@b_conv1*&
_output_shapes
:
Ą
 b_conv1/Adam_1/Initializer/zerosConst*
_class
loc:@b_conv1*%
valueB*    *&
_output_shapes
:*
dtype0
Ž
b_conv1/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@b_conv1*&
_output_shapes
:*
shape:*
shared_name 
Ç
b_conv1/Adam_1/AssignAssignb_conv1/Adam_1 b_conv1/Adam_1/Initializer/zeros*&
_output_shapes
:*
validate_shape(*
_class
loc:@b_conv1*
T0*
use_locking(
|
b_conv1/Adam_1/readIdentityb_conv1/Adam_1*
T0*
_class
loc:@b_conv1*&
_output_shapes
:

w_conv2/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
: *
_class
loc:@w_conv2*%
valueB *    
Ź
w_conv2/Adam
VariableV2*
shape: *&
_output_shapes
: *
shared_name *
_class
loc:@w_conv2*
dtype0*
	container 
Á
w_conv2/Adam/AssignAssignw_conv2/Adamw_conv2/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*&
_output_shapes
: *
_class
loc:@w_conv2
x
w_conv2/Adam/readIdentityw_conv2/Adam*
T0*
_class
loc:@w_conv2*&
_output_shapes
: 
Ą
 w_conv2/Adam_1/Initializer/zerosConst*&
_output_shapes
: *
dtype0*
_class
loc:@w_conv2*%
valueB *    
Ž
w_conv2/Adam_1
VariableV2*
shape: *&
_output_shapes
: *
shared_name *
_class
loc:@w_conv2*
dtype0*
	container 
Ç
w_conv2/Adam_1/AssignAssignw_conv2/Adam_1 w_conv2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@w_conv2*
validate_shape(*&
_output_shapes
: 
|
w_conv2/Adam_1/readIdentityw_conv2/Adam_1*
T0*&
_output_shapes
: *
_class
loc:@w_conv2

b_conv2/Adam/Initializer/zerosConst*
_class
loc:@b_conv2*%
valueB *    *&
_output_shapes
: *
dtype0
Ź
b_conv2/Adam
VariableV2*
	container *
dtype0*
_class
loc:@b_conv2*&
_output_shapes
: *
shape: *
shared_name 
Á
b_conv2/Adam/AssignAssignb_conv2/Adamb_conv2/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*&
_output_shapes
: *
_class
loc:@b_conv2
x
b_conv2/Adam/readIdentityb_conv2/Adam*
T0*&
_output_shapes
: *
_class
loc:@b_conv2
Ą
 b_conv2/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
: *
_class
loc:@b_conv2*%
valueB *    
Ž
b_conv2/Adam_1
VariableV2*
shared_name *
shape: *&
_output_shapes
: *
_class
loc:@b_conv2*
dtype0*
	container 
Ç
b_conv2/Adam_1/AssignAssignb_conv2/Adam_1 b_conv2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@b_conv2*
validate_shape(*&
_output_shapes
: 
|
b_conv2/Adam_1/readIdentityb_conv2/Adam_1*
_class
loc:@b_conv2*&
_output_shapes
: *
T0

 w_affine1/Adam/Initializer/zerosConst*
dtype0* 
_output_shapes
:
 *
_class
loc:@w_affine1*
valueB
 *    
¤
w_affine1/Adam
VariableV2*
shape:
 * 
_output_shapes
:
 *
shared_name *
_class
loc:@w_affine1*
dtype0*
	container 
Ă
w_affine1/Adam/AssignAssignw_affine1/Adam w_affine1/Adam/Initializer/zeros* 
_output_shapes
:
 *
validate_shape(*
_class
loc:@w_affine1*
T0*
use_locking(
x
w_affine1/Adam/readIdentityw_affine1/Adam*
T0* 
_output_shapes
:
 *
_class
loc:@w_affine1

"w_affine1/Adam_1/Initializer/zerosConst*
_class
loc:@w_affine1*
valueB
 *    *
dtype0* 
_output_shapes
:
 
Ś
w_affine1/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@w_affine1*
shared_name * 
_output_shapes
:
 *
shape:
 
É
w_affine1/Adam_1/AssignAssignw_affine1/Adam_1"w_affine1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@w_affine1*
validate_shape(* 
_output_shapes
:
 
|
w_affine1/Adam_1/readIdentityw_affine1/Adam_1* 
_output_shapes
:
 *
_class
loc:@w_affine1*
T0

 b_affine1/Adam/Initializer/zerosConst*
_class
loc:@b_affine1*
valueB*    *
_output_shapes	
:*
dtype0

b_affine1/Adam
VariableV2*
shape:*
_output_shapes	
:*
shared_name *
_class
loc:@b_affine1*
dtype0*
	container 
ž
b_affine1/Adam/AssignAssignb_affine1/Adam b_affine1/Adam/Initializer/zeros*
_class
loc:@b_affine1*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
s
b_affine1/Adam/readIdentityb_affine1/Adam*
T0*
_output_shapes	
:*
_class
loc:@b_affine1

"b_affine1/Adam_1/Initializer/zerosConst*
_class
loc:@b_affine1*
valueB*    *
_output_shapes	
:*
dtype0

b_affine1/Adam_1
VariableV2*
_output_shapes	
:*
dtype0*
shape:*
	container *
_class
loc:@b_affine1*
shared_name 
Ä
b_affine1/Adam_1/AssignAssignb_affine1/Adam_1"b_affine1/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@b_affine1
w
b_affine1/Adam_1/readIdentityb_affine1/Adam_1*
_output_shapes	
:*
_class
loc:@b_affine1*
T0

 w_affine2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	*
_class
loc:@w_affine2*
valueB	*    
˘
w_affine2/Adam
VariableV2*
	container *
dtype0*
_class
loc:@w_affine2*
_output_shapes
:	*
shape:	*
shared_name 
Â
w_affine2/Adam/AssignAssignw_affine2/Adam w_affine2/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*
_class
loc:@w_affine2
w
w_affine2/Adam/readIdentityw_affine2/Adam*
_class
loc:@w_affine2*
_output_shapes
:	*
T0

"w_affine2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	*
_class
loc:@w_affine2*
valueB	*    
¤
w_affine2/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:	*
_output_shapes
:	*
_class
loc:@w_affine2
Č
w_affine2/Adam_1/AssignAssignw_affine2/Adam_1"w_affine2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@w_affine2*
validate_shape(*
_output_shapes
:	
{
w_affine2/Adam_1/readIdentityw_affine2/Adam_1*
T0*
_class
loc:@w_affine2*
_output_shapes
:	

 b_affine2/Adam/Initializer/zerosConst*
_class
loc:@b_affine2*
valueB*    *
_output_shapes
:*
dtype0

b_affine2/Adam
VariableV2*
shared_name *
_class
loc:@b_affine2*
	container *
shape:*
dtype0*
_output_shapes
:
˝
b_affine2/Adam/AssignAssignb_affine2/Adam b_affine2/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@b_affine2
r
b_affine2/Adam/readIdentityb_affine2/Adam*
T0*
_class
loc:@b_affine2*
_output_shapes
:

"b_affine2/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@b_affine2*
valueB*    

b_affine2/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@b_affine2*
shared_name *
_output_shapes
:*
shape:
Ă
b_affine2/Adam_1/AssignAssignb_affine2/Adam_1"b_affine2/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@b_affine2
v
b_affine2/Adam_1/readIdentityb_affine2/Adam_1*
_output_shapes
:*
_class
loc:@b_affine2*
T0
O

Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wž?
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Î
Adam/update_w_conv1/ApplyAdam	ApplyAdamw_conv1w_conv1/Adamw_conv1/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_nesterov( *
_class
loc:@w_conv1*
T0*
use_locking( 
Ë
Adam/update_b_conv1/ApplyAdam	ApplyAdamb_conv1b_conv1/Adamb_conv1/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@b_conv1*
use_nesterov( *&
_output_shapes
:
Đ
Adam/update_w_conv2/ApplyAdam	ApplyAdamw_conv2w_conv2/Adamw_conv2/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
_class
loc:@w_conv2*&
_output_shapes
: *
T0*
use_nesterov( *
use_locking( 
Í
Adam/update_b_conv2/ApplyAdam	ApplyAdamb_conv2b_conv2/Adamb_conv2/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
_class
loc:@b_conv2*&
_output_shapes
: *
T0*
use_nesterov( *
use_locking( 
Ň
Adam/update_w_affine1/ApplyAdam	ApplyAdam	w_affine1w_affine1/Adamw_affine1/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
 *
use_nesterov( *
_class
loc:@w_affine1*
T0*
use_locking( 
Ě
Adam/update_b_affine1/ApplyAdam	ApplyAdam	b_affine1b_affine1/Adamb_affine1/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_nesterov( *
_class
loc:@b_affine1*
T0*
use_locking( 
Ó
Adam/update_w_affine2/ApplyAdam	ApplyAdam	w_affine2w_affine2/Adamw_affine2/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_output_shapes
:	*
_class
loc:@w_affine2
Ë
Adam/update_b_affine2/ApplyAdam	ApplyAdam	b_affine2b_affine2/Adamb_affine2/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_output_shapes
:*
_class
loc:@b_affine2
ô
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_w_conv1/ApplyAdam^Adam/update_b_conv1/ApplyAdam^Adam/update_w_conv2/ApplyAdam^Adam/update_b_conv2/ApplyAdam ^Adam/update_w_affine1/ApplyAdam ^Adam/update_b_affine1/ApplyAdam ^Adam/update_w_affine2/ApplyAdam ^Adam/update_b_affine2/ApplyAdam*
_output_shapes
: *
_class
loc:@b_affine1*
T0

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@b_affine1
ö

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_w_conv1/ApplyAdam^Adam/update_b_conv1/ApplyAdam^Adam/update_w_conv2/ApplyAdam^Adam/update_b_conv2/ApplyAdam ^Adam/update_w_affine1/ApplyAdam ^Adam/update_b_affine1/ApplyAdam ^Adam/update_w_affine2/ApplyAdam ^Adam/update_b_affine2/ApplyAdam*
_output_shapes
: *
_class
loc:@b_affine1*
T0

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@b_affine1*
validate_shape(*
_output_shapes
: 
˛
AdamNoOp^Adam/update_w_conv1/ApplyAdam^Adam/update_b_conv1/ApplyAdam^Adam/update_w_conv2/ApplyAdam^Adam/update_b_conv2/ApplyAdam ^Adam/update_w_affine1/ApplyAdam ^Adam/update_b_affine1/ApplyAdam ^Adam/update_w_affine2/ApplyAdam ^Adam/update_b_affine2/ApplyAdam^Adam/Assign^Adam/Assign_1
Ô
initNoOp^bias/Assign^w_conv1/Assign^b_conv1/Assign^w_conv2/Assign^b_conv2/Assign^w_affine1/Assign^b_affine1/Assign^w_affine2/Assign^b_affine2/Assign^beta1_power/Assign^beta2_power/Assign^w_conv1/Adam/Assign^w_conv1/Adam_1/Assign^b_conv1/Adam/Assign^b_conv1/Adam_1/Assign^w_conv2/Adam/Assign^w_conv2/Adam_1/Assign^b_conv2/Adam/Assign^b_conv2/Adam_1/Assign^w_affine1/Adam/Assign^w_affine1/Adam_1/Assign^b_affine1/Adam/Assign^b_affine1/Adam_1/Assign^w_affine2/Adam/Assign^w_affine2/Adam_1/Assign^b_affine2/Adam/Assign^b_affine2/Adam_1/Assign
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_c218fb2552774419b6cc98f5f27d579b/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ň
save/SaveV2/tensor_namesConst*
valueűBřB	b_affine1Bb_affine1/AdamBb_affine1/Adam_1B	b_affine2Bb_affine2/AdamBb_affine2/Adam_1Bb_conv1Bb_conv1/AdamBb_conv1/Adam_1Bb_conv2Bb_conv2/AdamBb_conv2/Adam_1Bbeta1_powerBbeta2_powerBbiasB	w_affine1Bw_affine1/AdamBw_affine1/Adam_1B	w_affine2Bw_affine2/AdamBw_affine2/Adam_1Bw_conv1Bw_conv1/AdamBw_conv1/Adam_1Bw_conv2Bw_conv2/AdamBw_conv2/Adam_1*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ţ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices	b_affine1b_affine1/Adamb_affine1/Adam_1	b_affine2b_affine2/Adamb_affine2/Adam_1b_conv1b_conv1/Adamb_conv1/Adam_1b_conv2b_conv2/Adamb_conv2/Adam_1beta1_powerbeta2_powerbias	w_affine1w_affine1/Adamw_affine1/Adam_1	w_affine2w_affine2/Adamw_affine2/Adam_1w_conv1w_conv1/Adamw_conv1/Adam_1w_conv2w_conv2/Adamw_conv2/Adam_1*)
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*
_output_shapes
:*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
_output_shapes
: *
T0
m
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBB	b_affine1
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/AssignAssign	b_affine1save/RestoreV2*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@b_affine1
t
save/RestoreV2_1/tensor_namesConst*#
valueBBb_affine1/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
Ś
save/Assign_1Assignb_affine1/Adamsave/RestoreV2_1*
_output_shapes	
:*
validate_shape(*
_class
loc:@b_affine1*
T0*
use_locking(
v
save/RestoreV2_2/tensor_namesConst*%
valueBBb_affine1/Adam_1*
_output_shapes
:*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
¨
save/Assign_2Assignb_affine1/Adam_1save/RestoreV2_2*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@b_affine1
o
save/RestoreV2_3/tensor_namesConst*
valueBB	b_affine2*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_3Assign	b_affine2save/RestoreV2_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@b_affine2
t
save/RestoreV2_4/tensor_namesConst*
dtype0*
_output_shapes
:*#
valueBBb_affine2/Adam
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Ľ
save/Assign_4Assignb_affine2/Adamsave/RestoreV2_4*
use_locking(*
T0*
_class
loc:@b_affine2*
validate_shape(*
_output_shapes
:
v
save/RestoreV2_5/tensor_namesConst*
dtype0*
_output_shapes
:*%
valueBBb_affine2/Adam_1
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
§
save/Assign_5Assignb_affine2/Adam_1save/RestoreV2_5*
use_locking(*
T0*
_class
loc:@b_affine2*
validate_shape(*
_output_shapes
:
m
save/RestoreV2_6/tensor_namesConst*
valueBBb_conv1*
_output_shapes
:*
dtype0
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_6Assignb_conv1save/RestoreV2_6*
_class
loc:@b_conv1*&
_output_shapes
:*
T0*
validate_shape(*
use_locking(
r
save/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:*!
valueBBb_conv1/Adam
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
­
save/Assign_7Assignb_conv1/Adamsave/RestoreV2_7*
_class
loc:@b_conv1*&
_output_shapes
:*
T0*
validate_shape(*
use_locking(
t
save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*
dtype0*#
valueBBb_conv1/Adam_1
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_8Assignb_conv1/Adam_1save/RestoreV2_8*
use_locking(*
T0*
_class
loc:@b_conv1*
validate_shape(*&
_output_shapes
:
m
save/RestoreV2_9/tensor_namesConst*
valueBBb_conv2*
_output_shapes
:*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_9Assignb_conv2save/RestoreV2_9*&
_output_shapes
: *
validate_shape(*
_class
loc:@b_conv2*
T0*
use_locking(
s
save/RestoreV2_10/tensor_namesConst*!
valueBBb_conv2/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_10Assignb_conv2/Adamsave/RestoreV2_10*
use_locking(*
validate_shape(*
T0*&
_output_shapes
: *
_class
loc:@b_conv2
u
save/RestoreV2_11/tensor_namesConst*#
valueBBb_conv2/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_11Assignb_conv2/Adam_1save/RestoreV2_11*&
_output_shapes
: *
validate_shape(*
_class
loc:@b_conv2*
T0*
use_locking(
r
save/RestoreV2_12/tensor_namesConst* 
valueBBbeta1_power*
_output_shapes
:*
dtype0
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_12Assignbeta1_powersave/RestoreV2_12*
_class
loc:@b_affine1*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
r
save/RestoreV2_13/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBbeta2_power
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_13Assignbeta2_powersave/RestoreV2_13*
_output_shapes
: *
validate_shape(*
_class
loc:@b_affine1*
T0*
use_locking(
k
save/RestoreV2_14/tensor_namesConst*
valueBBbias*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_14Assignbiassave/RestoreV2_14*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
	loc:@bias
p
save/RestoreV2_15/tensor_namesConst*
valueBB	w_affine1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
¨
save/Assign_15Assign	w_affine1save/RestoreV2_15*
_class
loc:@w_affine1* 
_output_shapes
:
 *
T0*
validate_shape(*
use_locking(
u
save/RestoreV2_16/tensor_namesConst*#
valueBBw_affine1/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save/Assign_16Assignw_affine1/Adamsave/RestoreV2_16*
_class
loc:@w_affine1* 
_output_shapes
:
 *
T0*
validate_shape(*
use_locking(
w
save/RestoreV2_17/tensor_namesConst*%
valueBBw_affine1/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_17Assignw_affine1/Adam_1save/RestoreV2_17* 
_output_shapes
:
 *
validate_shape(*
_class
loc:@w_affine1*
T0*
use_locking(
p
save/RestoreV2_18/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB	w_affine2
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
§
save/Assign_18Assign	w_affine2save/RestoreV2_18*
_class
loc:@w_affine2*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
u
save/RestoreV2_19/tensor_namesConst*
dtype0*
_output_shapes
:*#
valueBBw_affine2/Adam
k
"save/RestoreV2_19/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2
Ź
save/Assign_19Assignw_affine2/Adamsave/RestoreV2_19*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*
_class
loc:@w_affine2
w
save/RestoreV2_20/tensor_namesConst*
dtype0*
_output_shapes
:*%
valueBBw_affine2/Adam_1
k
"save/RestoreV2_20/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save/Assign_20Assignw_affine2/Adam_1save/RestoreV2_20*
_output_shapes
:	*
validate_shape(*
_class
loc:@w_affine2*
T0*
use_locking(
n
save/RestoreV2_21/tensor_namesConst*
valueBBw_conv1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_21/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
Ş
save/Assign_21Assignw_conv1save/RestoreV2_21*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:*
_class
loc:@w_conv1
s
save/RestoreV2_22/tensor_namesConst*
_output_shapes
:*
dtype0*!
valueBBw_conv1/Adam
k
"save/RestoreV2_22/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
_output_shapes
:*
dtypes
2
Ż
save/Assign_22Assignw_conv1/Adamsave/RestoreV2_22*
use_locking(*
T0*
_class
loc:@w_conv1*
validate_shape(*&
_output_shapes
:
u
save/RestoreV2_23/tensor_namesConst*
_output_shapes
:*
dtype0*#
valueBBw_conv1/Adam_1
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
_output_shapes
:*
dtypes
2
ą
save/Assign_23Assignw_conv1/Adam_1save/RestoreV2_23*&
_output_shapes
:*
validate_shape(*
_class
loc:@w_conv1*
T0*
use_locking(
n
save/RestoreV2_24/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBw_conv2
k
"save/RestoreV2_24/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
_output_shapes
:*
dtypes
2
Ş
save/Assign_24Assignw_conv2save/RestoreV2_24*
use_locking(*
T0*
_class
loc:@w_conv2*
validate_shape(*&
_output_shapes
: 
s
save/RestoreV2_25/tensor_namesConst*
dtype0*
_output_shapes
:*!
valueBBw_conv2/Adam
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
_output_shapes
:*
dtypes
2
Ż
save/Assign_25Assignw_conv2/Adamsave/RestoreV2_25*
_class
loc:@w_conv2*&
_output_shapes
: *
T0*
validate_shape(*
use_locking(
u
save/RestoreV2_26/tensor_namesConst*#
valueBBw_conv2/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_26Assignw_conv2/Adam_1save/RestoreV2_26*
use_locking(*
T0*
_class
loc:@w_conv2*
validate_shape(*&
_output_shapes
: 
Ů
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"ż
trainable_variables§¤
"
bias:0bias/Assignbias/read:0
+
	w_conv1:0w_conv1/Assignw_conv1/read:0
+
	b_conv1:0b_conv1/Assignb_conv1/read:0
+
	w_conv2:0w_conv2/Assignw_conv2/read:0
+
	b_conv2:0b_conv2/Assignb_conv2/read:0
1
w_affine1:0w_affine1/Assignw_affine1/read:0
1
b_affine1:0b_affine1/Assignb_affine1/read:0
1
w_affine2:0w_affine2/Assignw_affine2/read:0
1
b_affine2:0b_affine2/Assignb_affine2/read:0"
train_op

Adam"Ç
	variablesšś
"
bias:0bias/Assignbias/read:0
+
	w_conv1:0w_conv1/Assignw_conv1/read:0
+
	b_conv1:0b_conv1/Assignb_conv1/read:0
+
	w_conv2:0w_conv2/Assignw_conv2/read:0
+
	b_conv2:0b_conv2/Assignb_conv2/read:0
1
w_affine1:0w_affine1/Assignw_affine1/read:0
1
b_affine1:0b_affine1/Assignb_affine1/read:0
1
w_affine2:0w_affine2/Assignw_affine2/read:0
1
b_affine2:0b_affine2/Assignb_affine2/read:0
7
beta1_power:0beta1_power/Assignbeta1_power/read:0
7
beta2_power:0beta2_power/Assignbeta2_power/read:0
:
w_conv1/Adam:0w_conv1/Adam/Assignw_conv1/Adam/read:0
@
w_conv1/Adam_1:0w_conv1/Adam_1/Assignw_conv1/Adam_1/read:0
:
b_conv1/Adam:0b_conv1/Adam/Assignb_conv1/Adam/read:0
@
b_conv1/Adam_1:0b_conv1/Adam_1/Assignb_conv1/Adam_1/read:0
:
w_conv2/Adam:0w_conv2/Adam/Assignw_conv2/Adam/read:0
@
w_conv2/Adam_1:0w_conv2/Adam_1/Assignw_conv2/Adam_1/read:0
:
b_conv2/Adam:0b_conv2/Adam/Assignb_conv2/Adam/read:0
@
b_conv2/Adam_1:0b_conv2/Adam_1/Assignb_conv2/Adam_1/read:0
@
w_affine1/Adam:0w_affine1/Adam/Assignw_affine1/Adam/read:0
F
w_affine1/Adam_1:0w_affine1/Adam_1/Assignw_affine1/Adam_1/read:0
@
b_affine1/Adam:0b_affine1/Adam/Assignb_affine1/Adam/read:0
F
b_affine1/Adam_1:0b_affine1/Adam_1/Assignb_affine1/Adam_1/read:0
@
w_affine2/Adam:0w_affine2/Adam/Assignw_affine2/Adam/read:0
F
w_affine2/Adam_1:0w_affine2/Adam_1/Assignw_affine2/Adam_1/read:0
@
b_affine2/Adam:0b_affine2/Adam/Assignb_affine2/Adam/read:0
F
b_affine2/Adam_1:0b_affine2/Adam_1/Assignb_affine2/Adam_1/read:0*×
serving_defaultĂ
"
dropout
Placeholder_1:0
9
x_input.
Placeholder_2:0˙˙˙˙˙˙˙˙˙TT*
y_output
add_3:0˙˙˙˙˙˙˙˙˙

TEST_ROUND
bias:0e tensorflow/serving/predict