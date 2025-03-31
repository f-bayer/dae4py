#include <math.h>
#include <ida/ida.h> /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */

/* Problem Constants */
#define NEQ  2
#define NOUT 100

/* Prototypes of functions called by IDA */
int res(sunrealtype tres, N_Vector yy, N_Vector yp, N_Vector resval,
        void* user_data);

/* Prototypes of private functions */
static void PrintHeader(sunrealtype rtol, N_Vector avtol, N_Vector y, N_Vector yp);
static void PrintOutput(void* mem, sunrealtype t, N_Vector y);
static int check_retval(void* returnvalue, const char* funcname, int opt);
static int check_ans(N_Vector y, sunrealtype t, sunrealtype rtol, N_Vector atol);

/*
 *--------------------------------------------------------------------
 * Main Program
 *--------------------------------------------------------------------
 */
int main(void)
{
  void* mem;
  N_Vector yy, yp, avtol;
  sunrealtype rtol, *yval, *ypval, *atval;
  sunrealtype t0, tout1, tout, tret;
  int iout, retval, retvalr;
  int rootsfound[2];
  SUNMatrix A;
  SUNLinearSolver LS;
  SUNNonlinearSolver NLS;
  SUNContext ctx;
  FILE* FID;

  mem = NULL;
  yy = yp = avtol = NULL;
  yval = ypval = atval = NULL;
  A                    = NULL;
  LS                   = NULL;
  NLS                  = NULL;

  /* Create SUNDIALS context */
  retval = SUNContext_Create(SUN_COMM_NULL, &ctx);
  if (check_retval(&retval, "SUNContext_Create", 1)) { return (1); }

  /* Allocate N-vectors. */
  yy = N_VNew_Serial(NEQ, ctx);
  if (check_retval((void*)yy, "N_VNew_Serial", 0)) { return (1); }
  yp = N_VClone(yy);
  if (check_retval((void*)yp, "N_VNew_Serial", 0)) { return (1); }
  avtol = N_VClone(yy);
  if (check_retval((void*)avtol, "N_VNew_Serial", 0)) { return (1); }

  /* Create and initialize  y, y', and absolute tolerance vectors. */
  yval    = N_VGetArrayPointer(yy);
  yval[0] = 1.0;
  yval[1] = 0.0;

  ypval    = N_VGetArrayPointer(yp);
  ypval[0] = -1.0;
  ypval[1] = 1.0;

  rtol = 1.0e-6;
  atval    = N_VGetArrayPointer(avtol);
  atval[0] = 1.0e-6;
  atval[1] = 1.0e-6;

  /* Integration limits */
  t0    = 0.0;
  tout1 = 10.0;

  PrintHeader(rtol, avtol, yy, yp);

  /* Call IDACreate and IDAInit to initialize IDA memory */
  mem = IDACreate(ctx);
  if (check_retval((void*)mem, "IDACreate", 0)) { return (1); }
  retval = IDAInit(mem, res, t0, yy, yp);
  if (check_retval(&retval, "IDAInit", 1)) { return (1); }

  /* Call IDASetMaxNumSteps to set maximum number of steps */
  retval = IDASetMaxNumSteps(mem, 1000);
  if (check_retval(&retval, "IDASetMaxNumSteps", 1)) { return (1); }
  
  /* Call IDASVtolerances to set tolerances */
  retval = IDASVtolerances(mem, rtol, avtol);
  if (check_retval(&retval, "IDASVtolerances", 1)) { return (1); }

  /* Create dense SUNMatrix for use in linear solves */
  A = SUNDenseMatrix(NEQ, NEQ, ctx);
  if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return (1); }

  /* Create dense SUNLinearSolver object */
  LS = SUNLinSol_Dense(yy, A, ctx);
  if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return (1); }

  /* Attach the matrix and linear solver */
  retval = IDASetLinearSolver(mem, LS, A);
  if (check_retval(&retval, "IDASetLinearSolver", 1)) { return (1); }

  /* In loop, call IDASolve, print results, and test for error.
     Break out of loop when NOUT preset output times have been reached. */
  iout = 0;
  tout = tout1;
  while (1)
  {
    retval = IDASolve(mem, tout, &tret, yy, yp, IDA_NORMAL);

    PrintOutput(mem, tret, yy);

    if (check_retval(&retval, "IDASolve", 1)) { return (1); }

    if (retval == IDA_SUCCESS)
    {
      iout++;
      tout += 10.0;
    }

    if (iout == NOUT) { break; }
  }

  /* Print final statistics to the screen */
  printf("\nFinal Statistics:\n");
  retval = IDAPrintAllStats(mem, stdout, SUN_OUTPUTFORMAT_TABLE);

  /* check the solution error */
  retval = check_ans(yy, tret, rtol, avtol);

  /* Free memory */
  IDAFree(&mem);
  SUNNonlinSolFree(NLS);
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(avtol);
  N_VDestroy(yy);
  N_VDestroy(yp);
  SUNContext_Free(&ctx);

  return (retval);
}

/*
 * Define the system residual function.
 */
int res(sunrealtype tres, N_Vector yy, N_Vector yp, N_Vector rr,
        void* user_data)
{
  sunrealtype *yval, *ypval, *rval;

  yval  = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);
  rval  = N_VGetArrayPointer(rr);

  rval[0] = ypval[0] - tres * ypval[1] + yval[0] - (1.0 + tres) * yval[1];
  rval[1] = yval[1] - sin(tres);

  return (0);
}

/*
 * Print first lines of output (problem description)
 */
static void PrintHeader(sunrealtype rtol, N_Vector avtol, N_Vector y, N_Vector yp)
{
  sunrealtype *atval, *yval, *ypval;

  atval = N_VGetArrayPointer(avtol);
  yval  = N_VGetArrayPointer(y);
  ypval  = N_VGetArrayPointer(yp);

  printf("brenan_ida: Brenan's problem solved with IDA\n\n");
  printf("Linear solver: DENSE, with user-supplied Jacobian.\n");
  printf("Tolerance parameters:  rtol = %g   atol = %g %g \n", rtol,
         atval[0], atval[1]);
  printf("Initial conditions y0  = (%g %g)\n", yval[0], yval[1]);
  printf("Initial conditions yp0 = (%g %g)\n\n", ypval[0], ypval[1]);
  printf("---------------------------------------------------------------------"
         "--\n");
  printf("  t             y1           y2      | nst  k      h\n");
  printf("---------------------------------------------------------------------"
         "--\n");
}

/*
 * Print Output
 */
static void PrintOutput(void* mem, sunrealtype t, N_Vector y)
{
  sunrealtype* yval;
  int retval, kused;
  long int nst;
  sunrealtype hused;

  yval = N_VGetArrayPointer(y);

  retval = IDAGetLastOrder(mem, &kused);
  check_retval(&retval, "IDAGetLastOrder", 1);
  retval = IDAGetNumSteps(mem, &nst);
  check_retval(&retval, "IDAGetNumSteps", 1);
  retval = IDAGetLastStep(mem, &hused);
  check_retval(&retval, "IDAGetLastStep", 1);
  printf("%10.4e %12.4e %12.4e | %3ld  %1d %12.4e\n", t, yval[0],
         yval[1], nst, kused, hused);
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */
static int check_retval(void* returnvalue, const char* funcname, int opt)
{
  int* retval;
  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }
  else if (opt == 1)
  {
    /* Check if retval < 0 */
    retval = (int*)returnvalue;
    if (*retval < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return (1);
    }
  }
  else if (opt == 2 && returnvalue == NULL)
  {
    /* Check if function returned NULL pointer - no memory allocated */
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  return (0);
}

/* compare the solution at the final time to the exact solution*/
static int check_ans(N_Vector y, sunrealtype t, sunrealtype rtol, N_Vector atol)
{
  int passfail = 0; /* answer pass (0) or fail (1) retval */
  N_Vector ref;     /* reference solution vector        */
  N_Vector ewt;     /* error weight vector              */
  sunrealtype err;  /* wrms error                       */

  /* create reference solution and error weight vectors */
  ref = N_VClone(y);
  ewt = N_VClone(y);

  /* set the reference solution data */
  NV_Ith_S(ref, 0) = exp(-t) + t * sin(t);
  NV_Ith_S(ref, 1) = sin(t);

  /* compute the error weight vector, loosen atol */
  N_VAbs(ref, ewt);
  N_VLinearSum(rtol, ewt, SUN_RCONST(10.0), atol, ewt);
  if (N_VMin(ewt) <= 0.0)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: check_ans failed - ewt <= 0\n\n");
    return (-1);
  }
  N_VInv(ewt, ewt);

  /* compute the solution error */
  N_VLinearSum(1.0, y, -1.0, ref, ref);
  err = N_VWrmsNorm(ref, ewt);

  /* is the solution within the tolerances? */
  passfail = (err < 1.0) ? 0 : 1;

  if (passfail)
  {
    fprintf(stdout, "\nSUNDIALS_WARNING: check_ans error=%" "g" "\n\n", err);
  }

  /* Free vectors */
  N_VDestroy(ref);
  N_VDestroy(ewt);

  return (passfail);
}
