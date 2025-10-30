import java.util.Arrays;
import java.util.stream.*;
import java.util.Random;
import java.io.*;


public class Main {
    


    public static Random generator = new Random();

    public static double[][] rps = {
        {0, -1, 1, 0.2},
        {1, 0, -1, 0.2},
        {-1, 1, 0, 0.2},
        {0.1,0.1,0.1,0}
    }; 

    public static void main(String args[]){

        long startTime = System.nanoTime();


        for (int i =0 ; i< 1000; i++){
            int[] population = {5000, 5000, 5000, 5000};

            int iterations = 100000;

   
            double[][] results = moranSimulation(
                rps, 
                20000, 
                population, 
                iterations);
        }

        long endTime = System.nanoTime();

        long executionTime
            = (endTime - startTime) / 1000000;

        System.out.println("Simulation took for 1000 repeats"
                           + executionTime + "ms");


        //System.out.println(results.length);
        //System.out.println(results[0].length);
        //System.out.println(Arrays.toString(results[1]));

    }



    public static double[] payoffsAgainstPop(int[] population, double[][] matrix, int popSize){

        double[] payoffs = new double[4];

        for (int i = 0; i < 4; i++){
            double total = 0d;

            for (int j = 0; j < 4; j++){
                if (i == j){
                    total += (population[j] - 1) * matrix[i][j];
                } else{
                    total += population[j] * matrix[i][j];
                }
            }
            payoffs[i] = total / (popSize - 1);
        }


        return payoffs;
    }


    public static double[] moranSelection(double[] payoffs, double avg, int[] population, int popSize){


        double[] probs = new double[4];

        for (int i = 0; i < 4; i++){

            probs[i] = (population[i] * payoffs[i]) / (double) (popSize * avg);
        }


        return probs;
    }




    public static int weightedChoice(double[] weights){

        double total = DoubleStream.of(weights).sum();

        double r = generator.nextDouble() * total;

        double acc = 0d;

        for (int i =0; i < 4; i++){
            acc += weights[i];
            if (acc >= r){
                return i;
            }
        }
        return 3;
    }


    public static double[][] moranSimulation(double[][] matrix, int popSize, int[] population, int iterations){

        double w = 0.4;

        double[][] results = new double[4][iterations];

        for (int i =0 ; i < iterations; i++){

            int killed = weightedChoice(arrayDiv(population, popSize));

            double[] payoffs = payoffsAgainstPop(population, matrix, popSize);

            double[] p = pHelper(payoffs, w);

            double average = avg(p, population, popSize);

            double[] probs = moranSelection(payoffs, average, population, popSize);

            int chosen = weightedChoice(probs);

            population[chosen] += 1;
            population[killed] -= 1;

            for (int j = 0; j < 4; j++){
                results[j][i] = population[j] / (double) popSize;
            }
        }

        return results;
    
    }

    public static double avg(double[] p, int[] pop, int popSize){

        double avg = 0d;

        for (int i =0; i< 4; i++){
            avg += p[i] * pop[i];
        }

        return avg / (double) popSize;
    }


    public static double[] pHelper(double[] p, double w){
        double[] res = new double[4];
        for (int i = 0; i < 4; i++){
            res[i] = 1 - w + w * p[i];
        }
        return res;
    }

    public static double[] arrayDiv(int[] a, double x){
        double[] res = new double[4];

        for (int i = 0; i < 4; i++){
            res[i] = a[i] / x;
        }

        return res;
    }
}
