from dotenv import load_dotenv
from src.workflow import Workflow

load_dotenv()

def main():
    workflow = Workflow()
    print("Developer Tools Research Agent")

    while True:
        query = input("Developer Tools Quert: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if query:
            result = workflow.run(query)
            print(f"Results for: {query}")
            print("=" * 60)

            for i, company in enumerate(result.companies,1):
                print(f"{i} {company.name}")
                print(f" Website: {company.website}")
                print(f"Pricing: {company.pricing_model}")
                print(f"open source: {company.is_open_source}")

                if company.tech_stack:
                    print(f" Tech Stack: {', '.join(company.tech_stack[:5])}")

                if company.language_support:
                    print(f"Language support: {', '.join(company.language_support[:5])}")

                if company.api_available is not None:
                    api_status = ("Availale" if company.api_available else "Not Available")
                    print(f"API: {api_status}")

                if company.integration_capabilities:
                    print(f"Integrations: {', '.join(company.integration_capablities[:5])}")

                if company.description and company.description != "Failed":
                    print(f"Description: {company.description}")

                print()
            if result.analysis:
                print("Developer Recommendations: ")
                print("-" * 40)
                print(result.analysis)

if __name__ == "__main__":
    main()