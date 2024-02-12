/* =============================================================================================
 *
 *  compute_mingroc.cpp
 *  
 *  An optimization procedure to calculate the minimum information
 *  constant growth pattern connecting an initial 3D configuration
 *  with a final 3D configuration
 *
 *  by Dillon Cislo
 *  01/24/2024
 *
 *  This is a MEX-file for MATLAB
 *  
 * ============================================================================================*/

#include "mex.h" // for MATLAB

#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <complex>
#include <cmath>

#include "../../include/MINGROC/MINGROC.h"
#include "../../include/MINGROC/MINGROCParam.h"

// Main function
void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[] ) {

  typedef Eigen::MatrixXd MatrixXd;
  typedef Eigen::VectorXd VectorXd;
  typedef Eigen::MatrixXi MatrixXi;
  typedef Eigen::VectorXi VectorXi;

  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> ArrayXd;

  typedef std::complex<double> cdouble;
  typedef Eigen::Matrix<cdouble, Eigen::Dynamic, 1> CmplxVector;

  //-------------------------------------------------------------------------------------
  // INPUT PROCESSING
  //-------------------------------------------------------------------------------------

  // Check for proper number of arguments
  if ( nrhs != 9 ) {
    mexErrMsgIdAndTxt( "MATLAB:compute_mingroc:nargin",
        "COMPUTE_MINGROC requires 9 input arguments" );
  } else if ( nlhs != 4 ) {
    mexErrMsgIdAndTxt("MATLAB:compute_mingroc:nargout",
        "COMPUTE_MINGROC requries 4 output arguments" );
  }

  // Face connectivity list
  double *fIn = mxGetPr( prhs[0] );
  int numF = (int) mxGetM( prhs[0] );

  // 3D surface vertex coordinates
  double *vIn = mxGetPr( prhs[1] );
  int numV = (int) mxGetM( prhs[1] );

  // 2D pullback vertex coordinates
  double *xIn = mxGetPr( prhs[2] );

  // The Beltrami coefficient
  double *muIn = mxGetPr( prhs[5] );

  // The mapping coordinates
  double *wIn = mxGetPr( prhs[6] );

  // The final 3D surface coordinates
  double *fMapIn = mxGetPr( prhs[7] );

  // IDs of fixed vertices during optimization
  double *fixIn = mxGetPr( prhs[8] );
  int numFixed = (int) mxGetM( prhs[8] );

  // Map input arrays to Eigen-style matrix
  MatrixXd V = Eigen::Map<MatrixXd>(vIn, numV, 3);
  MatrixXd x = Eigen::Map<MatrixXd>(xIn, numV, 2);
  MatrixXd finMap3D = Eigen::Map<MatrixXd>(fMapIn, numV, 3);

  MatrixXd Fd = Eigen::Map<MatrixXd>(fIn, numF, 3);
  MatrixXi F = Fd.cast <int> ();
  F = (F.array() - 1).matrix(); // Account for MATLAB indexing

  VectorXd fixD = Eigen::Map<VectorXd>(fixIn, numFixed, 1);
  VectorXi fixIDx = fixD.cast <int> ();
  fixIDx = (fixIDx.array() - 1).matrix(); // Account for MATLAB indexing

  MatrixXd initMuR = Eigen::Map<MatrixXd>(muIn, numV, 2);
  CmplxVector initMu(numV, 1);
  initMu.real() = initMuR.col(0);
  initMu.imag() = initMuR.col(1);

  MatrixXd initWR = Eigen::Map<MatrixXd>(wIn, numV, 2);
  CmplxVector initW(numV, 1);
  initW.real() = initWR.col(0);
  initW.imag() = initWR.col(1);

  // Process MINGROC Input Options ------------------------------------------------------
  
  MINGROCpp::MINGROCParam<double> mingrocParam;

  int idx, tmp;

  // The size of the history used to approximate the inverse Hessian matrix
  if ( (idx = mxGetFieldNumber( prhs[3], "m" )) != -1 ) {
    mingrocParam.m = (int) *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The absolute tolerance for convergence test
  if ( (idx = mxGetFieldNumber( prhs[3], "epsilon" )) != -1 ) {
    mingrocParam.epsilon = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The relative tolerance for convergence test
  if ( (idx = mxGetFieldNumber( prhs[3], "epsilonRel" )) != -1 ) {
    mingrocParam.epsilonRel = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The distance for delta-based convergence test
  if ( (idx = mxGetFieldNumber( prhs[3], "past" )) != -1 ) {
    mingrocParam.past = (int) *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // Delta for convergence test
  if ( (idx = mxGetFieldNumber( prhs[3], "delta" )) != -1 ) {
    mingrocParam.delta = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The maximum number of iterations
  if ( (idx = mxGetFieldNumber( prhs[3], "maxIterations" )) != -1 ) {
    mingrocParam.maxIterations = (int) *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }
  
  // The minimization method
  if ( (idx = mxGetFieldNumber( prhs[3], "minimizationMethod" )) != -1 ) {
    mingrocParam.minimizationMethod = (int) *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The number of growth iterations for the alternating minimization method
  if ( (idx = mxGetFieldNumber( prhs[3], "numGrowthIterations" )) != -1 ) {
    mingrocParam.numGrowthIterations = (int) *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The number of Beltrami iterations for the alternating minimization method
  if ( (idx = mxGetFieldNumber( prhs[3], "numMuIterations" )) != -1 ) {
    mingrocParam.numMuIterations = (int) *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The smoothing time coefficient
  if ( (idx = mxGetFieldNumber( prhs[3], "tCoef" )) != -1 ) {
    mingrocParam.tCoef = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The line search termination condition
  if ( (idx = mxGetFieldNumber( prhs[3], "lineSearchTermination" )) != -1 ) {
    mingrocParam.lineSearchTermination = (int) *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The line search method
  if ( (idx = mxGetFieldNumber( prhs[3], "lineSearchMethod" )) != -1 ) {
    mingrocParam.lineSearchMethod = (int) *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The maximum number of trials for the line search
  if ( (idx = mxGetFieldNumber( prhs[3], "maxLineSearch" )) != -1 ) {
    mingrocParam.maxLineSearch = (int) *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The minimum step length allowed in the line search
  if ( (idx = mxGetFieldNumber( prhs[3], "minStep" )) != -1 ) {
    mingrocParam.minStep = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The maximum step length allowed in the line search
  if ( (idx = mxGetFieldNumber( prhs[3], "maxStep" )) != -1 ) {
    mingrocParam.maxStep = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // A parameter to control the accuray of the line search routine
  if ( (idx = mxGetFieldNumber( prhs[3], "ftol" )) != -1 ) {
    mingrocParam.ftol = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The coefficient for the Wolfe condition
  if ( (idx = mxGetFieldNumber( prhs[3], "wolfe" )) != -1 ) {
    mingrocParam.wolfe = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The conformality coefficient
  if ( (idx = mxGetFieldNumber( prhs[3], "CC" )) != -1 ) {
    mingrocParam.CC = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The smoothness coefficient
  if ( (idx = mxGetFieldNumber( prhs[3], "SC" )) != -1 ) {
    mingrocParam.SC = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // The diffeomorphism coefficient
  if ( (idx = mxGetFieldNumber( prhs[3], "DC" )) != -1 ) {
    mingrocParam.DC = *mxGetPr(mxGetFieldByNumber( prhs[3], 0, idx ));
  }

  // Display option for text output
  if ( (idx = mxGetFieldNumber( prhs[3], "iterDisp" )) != -1 ) {
    mingrocParam.iterDisp = *mxGetLogicals(mxGetFieldByNumber( prhs[3], 0, idx));
  }

  // Display option for detailed text output
  if ( (idx = mxGetFieldNumber( prhs[3], "iterDispDetailed" )) != -1 ) {
    mingrocParam.iterDispDetailed = *mxGetLogicals(mxGetFieldByNumber( prhs[3], 0, idx));
  }

  // Display option for visual output
  if ( (idx = mxGetFieldNumber( prhs[3], "iterVisDisp" )) != -1 ) {
    mingrocParam.iterVisDisp = *mxGetLogicals(mxGetFieldByNumber( prhs[3], 0, idx));
  }

  // Check option for mesh self-intersection checks in the virtual isothermal
  // parameterization
  if ( (idx = mxGetFieldNumber( prhs[3], "checkSelfIntersections" )) != -1 ) {
    mingrocParam.checkSelfIntersections =
      *mxGetLogicals(mxGetFieldByNumber( prhs[3], 0, idx));
  }

  // Iterative update option for Beltrami coefficient
  if ( (idx = mxGetFieldNumber( prhs[3], "recomputeMu" )) != -1 ) {
    mingrocParam.recomputeMu = *mxGetLogicals(mxGetFieldByNumber( prhs[3], 0, idx));
  }

  // Option to smooth Beltrami coefficient on final surface in alternating scheme
  if ( (idx = mxGetFieldNumber( prhs[3], "smoothMuOnFinalSurface" )) != -1 ) {
    mingrocParam.smoothMuOnFinalSurface =
      *mxGetLogicals(mxGetFieldByNumber( prhs[3], 0, idx));
  }

  // Option to calculate 2D or 3D areas/derivatives in the energy
  if ( (idx = mxGetFieldNumber( prhs[3], "use3DEnergy" )) != -1 ) {
    mingrocParam.use3DEnergy = *mxGetLogicals(mxGetFieldByNumber( prhs[3], 0, idx));
  }

  // Option to treat the Beltrami coefficient as a vector during smoothing
  if ( (idx = mxGetFieldNumber( prhs[3], "useVectorSmoothing" )) != -1 ) {
    mingrocParam.useVectorSmoothing =
      *mxGetLogicals(mxGetFieldByNumber( prhs[3], 0, idx));
  }

  // Option to treat the Beltrami coefficient as a vector during energy/
  // energy gradient computations
  if ( (idx = mxGetFieldNumber( prhs[3], "useVectorEnergy" )) != -1 ) {
    mingrocParam.useVectorEnergy =
      *mxGetLogicals(mxGetFieldByNumber( prhs[3], 0, idx));
  }

  

  // Process NNI Input Options ----------------------------------------------------------
  
  NNIpp::NNIParam<double> nniParam( numV, 3 );
  
  // The method used for constructing ghost points
  if ( (idx = mxGetFieldNumber( prhs[4], "ghostMethod" )) != -1 ) {
    nniParam.ghostMethod = (int) *mxGetPr(mxGetFieldByNumber( prhs[4], 0, idx ));
  }

  // Custom ghost point coordinates
  if ( (idx = mxGetFieldNumber( prhs[4], "customGhostPoints" )) != -1 ) {

    int GPn = (int) mxGetM(mxGetFieldByNumber( prhs[4], 0, idx ));

    nniParam.customGhostPoints =
      Eigen::Map<MatrixXd>( mxGetPr(mxGetFieldByNumber( prhs[4], 0, idx )), GPn, 1 );

    // Override other directives for ghost point construction
    nniParam.ghostMethod = NNIpp::NNI_GHOST_POINTS_CUSTOM;

  }

  // Edge length increase factor used for edge-based ghost point construction
  if ( (idx = mxGetFieldNumber( prhs[4], "GPe" )) != -1 ) {
    nniParam.GPe = *mxGetPr(mxGetFieldByNumber( prhs[4], 0, idx ));
  }

  // Radius increase factor of ghost point circle
  if ( (idx = mxGetFieldNumber( prhs[4], "GPr" )) != -1) {
    nniParam.GPr = *mxGetPr(mxGetFieldByNumber( prhs[4], 0, idx ));
  }

  // The number of ghost points to create using the dense circle method
  if ( (idx = mxGetFieldNumber( prhs[4], "GPn" )) != -1) {
    nniParam.GPn = (int) *mxGetPr(mxGetFieldByNumber( prhs[4], 0, idx ));
  }

  // The method used for gradient generation
  if ( (idx = mxGetFieldNumber( prhs[4], "gradType" )) != -1 ) {
    nniParam.gradType = (int) *mxGetPr(mxGetFieldByNumber( prhs[4], 0, idx ));
  }

  // Parameter for Sibson's method for gradient generation
  if ( (idx = mxGetFieldNumber( prhs[4], "iterAlpha" )) != -1 ) {
    nniParam.iterAlpha = *mxGetPr(mxGetFieldByNumber( prhs[4], 0, idx ));
  }

  // Display option for gradient generation progress
  if ( (idx = mxGetFieldNumber( prhs[4], "dispGrad" )) != -1 ) {
    nniParam.dispGrad = *mxGetLogicals(mxGetFieldByNumber( prhs[4], 0, idx));
  }

  // Display option for interpolation progress
  if ( (idx= mxGetFieldNumber( prhs[4], "dispInterp" )) != -1 ) {
    nniParam.dispInterp = *mxGetLogicals(mxGetFieldByNumber( prhs[4], 0, idx));
  }

  int idX = mxGetFieldNumber( prhs[4], "DataGradX" );
  bool hasDX = idX != -1;

  int idY = mxGetFieldNumber( prhs[4], "DataGradY" );
  bool hasDY = idY != -1;

  if ( hasDX && hasDY ) {

    nniParam.DataGradX =
      Eigen::Map<MatrixXd>(mxGetPr(mxGetFieldByNumber(prhs[4], 0, idX)), numV, 3);

    nniParam.DataGradY =
      Eigen::Map<MatrixXd>(mxGetPr(mxGetFieldByNumber(prhs[4], 0, idY)), numV, 3);

    nniParam.gradientSupplied = true;

  } else if ( hasDX || hasDY ) {

    mexErrMsgTxt("Incomplete gradient data supplied");
  
  }

  int idXX = mxGetFieldNumber( prhs[4], "DataHessXX" );
  bool hasDXX = idXX != -1;

  int idXY = mxGetFieldNumber( prhs[4], "DataHessXY" );
  bool hasDXY = idXY != -1;

  int idYY = mxGetFieldNumber( prhs[4], "DataHessYY" );
  bool hasDYY = idYY != -1;

  if ( hasDXX && hasDXY && hasDYY ) {

    nniParam.DataHessXX =
      Eigen::Map<MatrixXd>(mxGetPr(mxGetFieldByNumber(prhs[4], 0, idXX)), numV, 3);

    nniParam.DataHessXY =
      Eigen::Map<MatrixXd>(mxGetPr(mxGetFieldByNumber(prhs[4], 0, idXY)), numV, 3);

    nniParam.DataHessYY =
      Eigen::Map<MatrixXd>(mxGetPr(mxGetFieldByNumber(prhs[4], 0, idYY)), numV, 3);

    nniParam.hessianSupplied = true;

  } else if ( hasDXX || hasDYY || hasDYY ) {

    mexErrMsgTxt("Incomplete Hessian data supplied!");

  }

  //-------------------------------------------------------------------------------------
  // RUN OPTIMIZATION
  //-------------------------------------------------------------------------------------

  if ( mingrocParam.useVectorEnergy )
    mexErrMsgTxt("Vector energy is not yet implemented!");

  if ( mingrocParam.useVectorSmoothing )
    mexErrMsgTxt("Vector smoothing is not yet implemented!");

  // Generate a MINGROC object
  MINGROCpp::MINGROC<double, int> mingroc(F, V, x, mingrocParam, nniParam);

  double E = 0.0;
  CmplxVector mu = initMu;
  CmplxVector w = initW;
  MatrixXd map3D = finMap3D;

  try {

    mingroc( finMap3D, initMu, initW, fixIDx, E, mu, w, map3D );

  } catch (const std::exception &e) {

    mexWarnMsgTxt(e.what());

  }

  //-------------------------------------------------------------------------------------
  // OUTPUT PROCESSING
  //-------------------------------------------------------------------------------------

  plhs[0] = mxCreateDoubleScalar(E);

  plhs[1] = mxCreateDoubleMatrix(numV, 1, mxCOMPLEX);
  mxComplexDouble *muOut = mxGetComplexDoubles(plhs[1]);

  plhs[2] = mxCreateDoubleMatrix(numV, 1, mxCOMPLEX);
  mxComplexDouble *wOut = mxGetComplexDoubles(plhs[2]);

  for( int i = 0; i < numV; i++ )
  {
    muOut[i].real = mu(i).real();
    muOut[i].imag = mu(i).imag();

    wOut[i].real = w(i).real();
    wOut[i].imag = w(i).imag();
  }

  plhs[3] = mxCreateDoubleMatrix( numV, 3, mxREAL );
  Eigen::Map<MatrixXd>(mxGetPr(plhs[3]), numV, 3) = map3D;

  return;

};
