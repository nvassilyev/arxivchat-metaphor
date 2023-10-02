from utils import conversational_qa, search_arxiv
import json

with open("prompts.json", 'r') as json_file:
    prompts = json.load(json_file)

def main():
    """Main function for the CLI."""
    total_cost = 0
    print(f"\n{prompts['welcome']}\n")

    while True:
        print(prompts["search_instruction"], '\n')
        query = input("USER (SEARCH): ")
        summarized, rank = search_arxiv(query)
        
        print(f"\n{prompts['choose_paper']}\n{prompts['commands']}")

        for i, a_id in enumerate(summarized.keys()):
            print(f"{i+1}. {summarized[a_id]['title']}")

        action_list = prompts["action_list"]
        action = input("\nUSER (CHOOSE PAPER ID): ")

        while action not in action_list:
            print("\nInvalid input, please try again.\n")
            action = input("USER (CHOOSE PAPER ID): ")
        if action == "q":
            break
        elif action == "s":
            continue
        else:
            if action:
                rank = int(action) - 1
            arxiv_id = list(summarized.keys())[rank]
            print(f"\nYou chose paper {rank+1}. {summarized[arxiv_id]['title']}\n")
            if conversational_qa(arxiv_id) == "q":
                break

    print(f"\n{prompts['exit']}")

if __name__ == "__main__":
    main()