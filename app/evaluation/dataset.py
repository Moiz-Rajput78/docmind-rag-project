# ============================================================
# DOCMIND — RETRIEVAL EVALUATION DATASET
# ============================================================

"""
30-question retrieval evaluation dataset.

Each question contains:
- question
- difficulty
- category
- expected_sources

expected_sources contains filenames that should be retrieved
for the question to count as a successful retrieval.
"""


EVALUATION_QUESTIONS = [

    # ========================================================
    # EASY — 10 QUESTIONS
    # ========================================================

    {
        "id": 1,
        "question": "How many annual leave days do full-time employees receive?",
        "difficulty": "easy",
        "category": "hr",
        "expected_sources": ["leave_policy.pdf"],
    },
    {
        "id": 2,
        "question": "How many sick leave days do employees receive?",
        "difficulty": "easy",
        "category": "hr",
        "expected_sources": ["leave_policy.pdf"],
    },
    {
        "id": 3,
        "question": "What is NovaDesk?",
        "difficulty": "easy",
        "category": "products",
        "expected_sources": ["product_a.pdf"],
    },
    {
        "id": 4,
        "question": "What is NovaFlow?",
        "difficulty": "easy",
        "category": "products",
        "expected_sources": ["product_b.txt"],
    },
    {
        "id": 5,
        "question": "What departments does NovaTech have?",
        "difficulty": "easy",
        "category": "company",
        "expected_sources": ["company_overview.pdf"],
    },
    {
        "id": 6,
        "question": "What working model does NovaTech use?",
        "difficulty": "easy",
        "category": "company",
        "expected_sources": ["company_overview.pdf"],
    },
    {
        "id": 7,
        "question": "What are the system requirements for NovaDesk?",
        "difficulty": "easy",
        "category": "technical",
        "expected_sources": ["installation.pdf"],
    },
    {
        "id": 8,
        "question": "How should employees record attendance?",
        "difficulty": "easy",
        "category": "hr",
        "expected_sources": ["attendance_policy.docx"],
    },
    {
        "id": 9,
        "question": "What ticket priorities does NovaDesk support?",
        "difficulty": "easy",
        "category": "products",
        "expected_sources": ["faq.txt"],
    },
    {
        "id": 10,
        "question": "What should I check if an application does not start?",
        "difficulty": "easy",
        "category": "technical",
        "expected_sources": ["troubleshooting.txt"],
    },


    # ========================================================
    # MEDIUM — 10 QUESTIONS
    # ========================================================

    {
        "id": 11,
        "question": "How far in advance should planned annual leave normally be requested?",
        "difficulty": "medium",
        "category": "hr",
        "expected_sources": ["leave_policy.pdf"],
    },
    {
        "id": 12,
        "question": "Who reviews employee leave requests?",
        "difficulty": "medium",
        "category": "hr",
        "expected_sources": ["leave_policy.pdf"],
    },
    {
        "id": 13,
        "question": "What types of customer-support activities can NovaDesk manage?",
        "difficulty": "medium",
        "category": "products",
        "expected_sources": ["product_a.pdf"],
    },
    {
        "id": 14,
        "question": "What kind of business processes is NovaFlow designed to automate?",
        "difficulty": "medium",
        "category": "products",
        "expected_sources": ["product_b.txt"],
    },
    {
        "id": 15,
        "question": "What services does NovaTech provide to customers?",
        "difficulty": "medium",
        "category": "company",
        "expected_sources": ["services.docx"],
    },
    {
        "id": 16,
        "question": "What happens during a typical NovaTech software development project?",
        "difficulty": "medium",
        "category": "company",
        "expected_sources": ["services.docx"],
    },
    {
        "id": 17,
        "question": "What areas can be configured in NovaDesk?",
        "difficulty": "medium",
        "category": "technical",
        "expected_sources": ["configuration.docx"],
    },
    {
        "id": 18,
        "question": "What should administrators verify after installing NovaDesk?",
        "difficulty": "medium",
        "category": "technical",
        "expected_sources": ["installation.pdf"],
    },
    {
        "id": 19,
        "question": "What should a user check when they cannot sign in?",
        "difficulty": "medium",
        "category": "technical",
        "expected_sources": ["troubleshooting.txt"],
    },
    {
        "id": 20,
        "question": "How should employees handle unexpected absence from work?",
        "difficulty": "medium",
        "category": "hr",
        "expected_sources": ["attendance_policy.docx"],
    },


    # ========================================================
    # DIFFICULT — 10 QUESTIONS
    # ========================================================

    {
        "id": 21,
        "question": "An employee wants to take planned personal time away from work. What type of leave should they use and how should they request it?",
        "difficulty": "difficult",
        "category": "hr",
        "expected_sources": ["leave_policy.pdf"],
    },
    {
        "id": 22,
        "question": "A full-time employee needs time off for both planned personal reasons and illness. What leave options are available and how many days are provided?",
        "difficulty": "difficult",
        "category": "hr",
        "expected_sources": ["leave_policy.pdf"],
    },
    {
        "id": 23,
        "question": "A support team wants to manage customer requests while tracking ticket status and service performance. Which NovaTech product fits this need?",
        "difficulty": "difficult",
        "category": "products",
        "expected_sources": ["product_a.pdf"],
    },
    {
        "id": 24,
        "question": "A company wants to automate repetitive business workflows rather than manage customer support tickets. Which NovaTech product should it consider?",
        "difficulty": "difficult",
        "category": "products",
        "expected_sources": ["product_b.txt"],
    },
    {
        "id": 25,
        "question": "A new NovaDesk installation needs both browser access and administrative configuration. What prerequisites and post-installation checks are required?",
        "difficulty": "difficult",
        "category": "technical",
        "expected_sources": ["installation.pdf"],
    },
    {
        "id": 26,
        "question": "NovaDesk is not starting after configuration changes. What troubleshooting steps should an administrator take?",
        "difficulty": "difficult",
        "category": "technical",
        "expected_sources": ["troubleshooting.txt", "configuration.docx"],
    },
    {
        "id": 27,
        "question": "An employee is working remotely. What attendance-related responsibilities do they have?",
        "difficulty": "difficult",
        "category": "hr",
        "expected_sources": ["attendance_policy.docx"],
    },
    {
        "id": 28,
        "question": "How does NovaTech's hybrid working model affect employees working remotely?",
        "difficulty": "difficult",
        "category": "company",
        "expected_sources": [
            "company_overview.pdf",
            "attendance_policy.docx",
        ],
    },
    {
        "id": 29,
        "question": "What security responsibilities apply when employees work remotely or handle company information?",
        "difficulty": "difficult",
        "category": "company",
        "expected_sources": [
            "policies.pdf",
            "company_overview.pdf",
        ],
    },
    {
        "id": 30,
        "question": "A NovaDesk user cannot sign in and the application also has configuration problems. Which documentation should be consulted and what areas should be checked?",
        "difficulty": "difficult",
        "category": "technical",
        "expected_sources": [
            "troubleshooting.txt",
            "configuration.docx",
        ],
    },
]


def get_evaluation_questions():
    """Return the complete evaluation dataset."""
    return EVALUATION_QUESTIONS


def get_questions_by_difficulty(difficulty: str):
    """Return questions for a specific difficulty level."""
    return [
        question
        for question in EVALUATION_QUESTIONS
        if question["difficulty"].lower() == difficulty.lower()
    ]


def get_questions_by_category(category: str):
    """Return questions for a specific category."""
    return [
        question
        for question in EVALUATION_QUESTIONS
        if question["category"].lower() == category.lower()
    ]


def validate_dataset():
    """Validate that the evaluation dataset meets project requirements."""

    assert len(EVALUATION_QUESTIONS) == 30, (
        f"Expected 30 questions, found {len(EVALUATION_QUESTIONS)}"
    )

    difficulties = {
        "easy": 0,
        "medium": 0,
        "difficult": 0,
    }

    for question in EVALUATION_QUESTIONS:
        difficulties[question["difficulty"]] += 1

        assert question["question"].strip()
        assert question["expected_sources"]

    assert difficulties["easy"] == 10
    assert difficulties["medium"] == 10
    assert difficulties["difficult"] == 10

    return True


if __name__ == "__main__":

    print("=" * 70)
    print("DOCMIND — EVALUATION DATASET")
    print("=" * 70)

    validate_dataset()

    print(f"\nTotal questions: {len(EVALUATION_QUESTIONS)}")
    print(f"Easy:  {len(get_questions_by_difficulty('easy'))}")
    print(f"Medium: {len(get_questions_by_difficulty('medium'))}")
    print(f"Difficult: {len(get_questions_by_difficulty('difficult'))}")

    print("\nDATASET VALIDATION PASSED")