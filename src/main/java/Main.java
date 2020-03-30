import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.genetics.AbstractListChromosome;
import org.apache.commons.math3.genetics.Chromosome;
import org.apache.commons.math3.genetics.ChromosomePair;
import org.apache.commons.math3.genetics.CrossoverPolicy;
import org.apache.commons.math3.genetics.ElitisticListPopulation;
import org.apache.commons.math3.genetics.FixedElapsedTime;
import org.apache.commons.math3.genetics.GeneticAlgorithm;
import org.apache.commons.math3.genetics.InvalidRepresentationException;
import org.apache.commons.math3.genetics.MutationPolicy;
import org.apache.commons.math3.genetics.Population;
import org.apache.commons.math3.genetics.TournamentSelection;

interface Function {
	double value(List<Double> input);
}

class SphereFunction implements Function {
	@Override
	public double value(List<Double> input) {
		double output = 0;

		for (Double value : input) {
			output += value * value;
		}

		return output;
	}
}

class StyblinskiTangFunction implements Function {
	@Override
	public double value(List<Double> input) {
		double output = 0;

		for (Double value : input) {
			output += value * value * value * value - 16 * value * value
					+ 5 * value;
		}

		return output / 2D;
	}
}

class RastriginFunction implements Function {
	@Override
	public double value(List<Double> input) {
		double output = 0;

		for (Double value : input) {
			output += value * value - 10 * Math.cos(2 * Math.PI * value);
		}

		output += 10 * input.size();

		return output;
	}
}

class RosenbrockFunction implements Function {
	@Override
	public double value(List<Double> input) {
		double output = 0;

		for (int i = 0; i < input.size() - 1; i++) {
			double xi0 = input.get(i);
			double xi1 = input.get(i + 1);

			output += 100 * (xi1 - xi0 * xi0) * (xi1 - xi0 * xi0)
					+ (1 - xi0) * (1 - xi0);
		}

		return output;
	}
}

class DoubleListChromosome extends AbstractListChromosome<Double> {
	private Function function = null;

	private DoubleListChromosome(List<Double> chromosome, Function function) {
		super(chromosome);
		this.function = function;
	}

	public DoubleListChromosome(Double[] chromosome, Function function)
			throws InvalidRepresentationException {
		super(chromosome);
		this.function = function;
	}

	Function getFunction() {
		return function;
	}

	List<Double> getValues() {
		return super.getRepresentation();
	}

	@Override
	public double fitness() {
		/* The most fitted chromosome has a minimum value. */
		return -function.value(getRepresentation());
	}

	@Override
	protected void checkValidity(List<Double> chromosome)
			throws InvalidRepresentationException {
	}

	@Override
	public AbstractListChromosome<Double> newFixedLengthChromosome(
			List<Double> chromosome) {
		return new DoubleListChromosome(chromosome, function);
	}
}

class UinformCrossover implements CrossoverPolicy {
	private static Random PRNG = new Random();

	@Override
	public ChromosomePair crossover(Chromosome first, Chromosome second)
			throws MathIllegalArgumentException {
		final List<Double> parent1 = ((DoubleListChromosome) first).getValues();
		final List<Double> parent2 = ((DoubleListChromosome) second)
				.getValues();

		final List<Double> child1 = new ArrayList<Double>();
		final List<Double> child2 = new ArrayList<Double>();

		for (int i = 0; i < parent1.size() && i < parent2.size(); i++) {
			if (PRNG.nextBoolean() == true) {
				child1.add(parent1.get(i));
				child2.add(parent2.get(i));
			} else {
				child1.add(parent2.get(i));
				child2.add(parent1.get(i));
			}
		}

		return new ChromosomePair(
				((DoubleListChromosome) first).newFixedLengthChromosome(child1),
				((DoubleListChromosome) second)
						.newFixedLengthChromosome(child2));
	}
}

class RandomDoubleMutation implements MutationPolicy {
	private static Random PRNG = new Random();

	@Override
	public Chromosome mutate(Chromosome original)
			throws MathIllegalArgumentException {
		List<Double> parent = ((DoubleListChromosome) original).getValues();

		Double values[] = new Double[parent.size()];

		for (int i = 0; i < values.length; i++) {
			values[i] = parent.get(i) + PRNG.nextDouble() - 0.5D;
		}

		return new DoubleListChromosome(values,
				((DoubleListChromosome) original).getFunction());
	}
}

public class Main {
	private static Random PRNG = new Random();

	private static int CHROMOSOME_SIZE = 10;

	private static int POPULATION_SIZE = 37;

	private static double ELITISM_RATE = 0.1;

	private static int TOURNAMENT_AIRITY = 2;

	private static double CROSSOVER_RATE = 0.9;

	private static double MUTATION_RATE = 0.01;

	private static int SINGLE_OPTIMIZATION_SECONDS = 60;

	private static int TOTOAL_NUMBER_OF_OPTIMIZATIONS = 10;

	public static void main(String[] args) {
		GeneticAlgorithm algorithm = new GeneticAlgorithm(
				new UinformCrossover(), CROSSOVER_RATE,
				new RandomDoubleMutation(), MUTATION_RATE,
				new TournamentSelection(TOURNAMENT_AIRITY));

		/* Select function for optimization. */
		Function function = new RosenbrockFunction();

		/* Initialize random solutions. */
		List<Chromosome> list = new ArrayList<Chromosome>();
		for (int i = 0; i < POPULATION_SIZE; i++) {
			Double values[] = new Double[CHROMOSOME_SIZE];
			for (int j = 0; j < values.length; j++) {
				values[j] = new Double(0.5D - PRNG.nextDouble());
			}
			list.add(new DoubleListChromosome(values, function));
		}

		/* Crate initial population. */
		Population initial, optimized;
		initial = optimized = new ElitisticListPopulation(list, list.size(),
				ELITISM_RATE);

		/* Optimize population. */
		for (int g = 0; g < TOTOAL_NUMBER_OF_OPTIMIZATIONS; g++) {
			System.out.println(optimized.getFittestChromosome());

			initial = optimized;
			optimized = algorithm.evolve(initial,
					new FixedElapsedTime(SINGLE_OPTIMIZATION_SECONDS));
		}

		System.out.println(optimized.getFittestChromosome());
	}

}
