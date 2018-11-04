#include <stdio.h>
#include <stdlib.h>

#define ROWS 20
#define TESTROWS 4
#define COLS 8

typedef struct _tab {
   int *x;   //Stores X values
   int result;  //Stores y value
} tab;

int Bayes_class(int *instance, tab *table) //Function to find aposteriori probability of each instance

{

   double py0 = 0, py1 = 0;   //py0 = P(y=0), py1 = P(y=1)
   double py_x0, py_x1;   //py_x0 = P(y=0|X), py_x1 = P(y=1|X)
   double *px_y0 = (double *) calloc(COLS, sizeof(double));  //px_y0[i] = P(Xi| y=0)
   double *px_y1 = (double *) calloc(COLS, sizeof(double));  //px_y1[i] = P(Xi| y=1)

   for(int i=0; i<ROWS; i++) {

      if(table[i].result == 0) py0 += 1;  //Counts training instances with y=0
   
      else if (table[i].result == 1) py1 += 1;  //Counts training instances with y=1

      for(int j=0; j<COLS; j++) {

         if(table[i].x[j] == instance[j] && table[i].result == 0) 
            px_y0[j] += 1;  //Counts training instances with X[j]=instance[j] and y=0

         else if (table[i].x[j] == instance[j] && table[i].result == 1)
            px_y1[j] += 1;  //Counts training instances with X[j]=instance[j] and y=1

      }

   }

   for(int j=0; j<COLS; j++) {
      px_y0[j] = (1+px_y0[j])/(2+py0);
      px_y1[j] = (1+px_y1[j])/(2+py1);
   }

   /*
   *The above loop converts counts into probabilities
   *Laplacian add 1 smoothing has been implemented by adding 1 to the numerator and 2 to the denominator (since each Xi can assume 2 values)
   */
   py0/= py0 + py1;
   py1/= py0 + py1;

   //py0 and py1 are converted into probabilities

   py_x0 = py0;
   py_x1 = py1;

   //Aposteriori probabilities are initialized to py0 and py1 respectively

   for(int j=0; j<COLS; j++) {

      py_x0 *= px_y0[j];
      
      py_x1 *= px_y1[j];

   }

   //Above loop calculates the non-normalized aposteriori probabilities

   return (py_x1 > py_x0);

}

int main()

{

   int i, j;

   tab *table = (tab *) malloc(ROWS*sizeof(tab));
   int **instance = (int **) malloc(TESTROWS*sizeof(int *)); 

   FILE *fp;

   fp = fopen("data3.csv", "r");

   if(fp==NULL) {
      printf("data3.csv not found!\n");
      return 1;
   }

   for (i=0; i<ROWS; i++) {

      table[i].x = (int *) malloc(COLS*sizeof(int));

      if(table[i].x == NULL) return 1;

      for (j=0; j<COLS; j++) {

         fscanf(fp, " %d,", &table[i].x[j]);

      }

      fscanf(fp, " %d,", &table[i].result);

   }

   fclose(fp);

   FILE *fp2 = fopen("test3.csv", "r");

   if(fp2==NULL) {
      printf("test3.csv not found!\n");
      return 1;
   }

   for (i=0; i<TESTROWS; i++) {

      instance[i] = (int *) malloc(COLS*sizeof(int));

      for(j=0; j<COLS; j++) {

         fscanf(fp2, " %d,", &instance[i][j]);

      }
 
   }

   fclose(fp2);

   FILE *fpw = fopen("Bayes_classifier.out", "w");

   if (fpw == NULL) {

      printf("Failed to create output file\n");
  
      for (i=0; i<TESTROWS; i++) 
        printf("%d ", Bayes_class(instance[i], table));

      printf("\n");

      return 1;

   }

   for(i=0; i<TESTROWS; i++) {
      fprintf(fp, "%d ", Bayes_class(instance[i], table));
      printf("%d ", Bayes_class(instance[i], table));
   }

   printf("\n");

   fclose(fpw);
   
   return 0;

}
