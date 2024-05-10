import pandas as pd
import argparse
import re


BENCHOPT_RUNNING_FORMAT = 'benchopt run -d "{}" -s {} --output "{}" {}'

# To make Benchopt parser happy
def enclose_with_brackets(text):
    # Define a regular expression pattern to find strings inside parentheses
    pattern = r'\((.*?)\)'

    # Define a function to add square brackets around matched strings
    def replace(match):
        return '[' + match.group(0) + ']'

    # Use re.sub() to replace matched strings with the modified version
    modified_text = re.sub(pattern, replace, text)

    return modified_text

def check_experiment_status(csv_file):
    df = pd.read_csv(csv_file)
    finished_experiments = df[df['Status'] == 'Finished']
    running_experiments = df[df['Status'] == 'Running']
    return finished_experiments, running_experiments

def generate_benchopt_commands(experiments, slurm_yaml=None):
    commands = []

    # Groupby per dataset
    grouped = experiments.groupby('Dataset')['Solver'].unique()

    for dataset, solvers in grouped.items():
        dataset = enclose_with_brackets(dataset)
        solver_string = " -s ".join(solvers)
    
        output_filename = f"output_{dataset}_{'_'.join(solvers)}"
        slurm_option = f"--slurm {slurm_yaml}" if slurm_yaml else ""

        commands.append(
            BENCHOPT_RUNNING_FORMAT.format(dataset, solver_string, output_filename, slurm_option)
        )

    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Benchopt commands for running or finished experiments.")
    parser.add_argument("csv_file", help="Path to the CSV file containing experiment status information.")
    parser.add_argument("--slurm", default=None, help="Path to the slurm yaml file.")
    args = parser.parse_args()

    csv_file = args.csv_file
    finished, running = check_experiment_status(csv_file)

    print("Finished experiments:")
    for idx, exp in finished.iterrows():
        print(exp.tolist())

    print("\nRunning experiments:")
    for idx, exp in running.iterrows():
        print(exp.tolist())


    finished_commands = generate_benchopt_commands(finished, args.slurm)
    finished_bash_file = "../run_finished_exps.sh"
    print("\nBenchopt commands for finished experiments:")
    with open(finished_bash_file, "w") as file:

        file.write('echo "Running finished experiments"\n')
        for cmd in finished_commands:
            print(cmd)
            file.write(cmd + "\n")
        
        file.write('\n\necho "All commands executed successfully"\n')


    running_commands = generate_benchopt_commands(running, args.slurm)
    running_bash_file = '../run_unfinished_exps.sh'
    print("\nBenchopt commands for running experiments:")
    with open(running_bash_file, "w") as file:
        for cmd in running_commands:
            print(cmd)
            file.write(cmd + "\n")
