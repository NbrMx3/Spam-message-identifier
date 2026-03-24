from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem


def build_pdf(output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2.0 * cm,
        leftMargin=2.0 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=16,
        leading=19,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading3"],
        fontSize=12,
        leading=14,
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        spaceAfter=5,
    )

    story = []

    story.append(Paragraph("Spam Message Identifier Using Machine Learning", title_style))
    story.append(Paragraph("University of Eastern Africa Baraton", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Well Defined Title of the Group's AI Project Problem", heading_style))
    story.append(
        Paragraph(
            "Developing an intelligent one-step spam message detection system that accurately distinguishes unsolicited SMS messages (spam) from legitimate personal/business messages (ham) in real time.",
            body_style,
        )
    )

    story.append(Paragraph("2. Clear Names of the Three Project Developers", heading_style))
    names = ListFlowable(
        [
            ListItem(Paragraph("Dennis Kipkemoi (SDENKI2314)", body_style)),
            ListItem(Paragraph("Baraka Kahindi (SBRIBA2211)", body_style)),
            ListItem(Paragraph("Lenard Kibet (SLENKI2311)", body_style)),
        ],
        bulletType="bullet",
        leftIndent=16,
    )
    story.append(names)

    story.append(Paragraph("3. Results and Discussion", heading_style))
    story.append(
        Paragraph(
            "The system was implemented with TF-IDF feature extraction (max_features=5000) and a Multinomial Naive Bayes classifier, then evaluated using an 80/20 train-test split and 5-fold cross-validation. On the current local dataset (5,572 messages), the model achieved cross-validation mean accuracy of <b>97.51%</b> (+/- 0.57%), test accuracy of <b>97.22%</b>, precision of <b>100.00%</b>, recall of <b>79.19%</b>, and F1-score of <b>88.39%</b>.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "These scores indicate the model is highly reliable when it flags a message as spam (very high precision and zero false positives in this split), which directly supports the objective of minimizing false spam alerts to users. However, the lower recall shows that some spam messages are still missed (31 false negatives), so the objective is only partially met for complete spam coverage. Overall, the model provides strong practical performance and is suitable for deployment, with clear room for recall improvement.",
            body_style,
        )
    )

    story.append(Paragraph("4. Summary and Conclusion", heading_style))
    story.append(
        Paragraph(
            "This project successfully delivered a functional spam classification pipeline and web interface that can classify single and batch messages through API endpoints and UI interaction. The achieved results confirm strong effectiveness, especially in precision and stability. For future improvement, the team should focus on increasing recall by tuning class priors/thresholds, testing alternative models (e.g., Linear SVM), applying richer text preprocessing (n-grams and lemmatization), and retraining on a larger and more diverse SMS corpus to improve generalization and reduce missed spam.",
            body_style,
        )
    )

    doc.build(story)


if __name__ == "__main__":
    build_pdf("AI_Project_One_Page_Report.pdf")
    print("Generated: AI_Project_One_Page_Report.pdf")
