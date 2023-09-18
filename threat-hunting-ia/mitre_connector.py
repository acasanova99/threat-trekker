"""
    mitre_connector.py

    This module retrieves all the necessary data from the mitre att&ck framework. It is useful to label the classes of
    the datasets.

    Author: Angel Casanova
    2023
"""
from pyattck import Attck


def main():
    print(f'start')
    attck = Attck()

    for technique in attck.enterprise.techniques:
        print(technique.name)
    print(f'end')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
