from __future__ import annotations

import colorsys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
NOTEBOOK_ROOT = ROOT / "Labs_notebooks"
ASSET_DIR = NOTEBOOK_ROOT / "assets"
FORMAL_ASSET_DIR = ASSET_DIR / "formal"
REFERENCE_ASSET_DIR = WORKSPACE / "midterm" / "Labs_notebooks" / "assets"
LOGO_SOURCE = WORKSPACE / "ai-nn-website" / "site" / "nirm1" / "report" / "AITU.png"

AUTHOR = "Sundetkhan Bekzat"
INSTRUCTOR = "Akhmetova Zhanar"
SUBJECT = "Artificial Intelligence and Neural Networks"


REFERENCE_IMAGES = {
    "course_completion_restyled.png": "course_completion.png",
    "regression_target_restyled.png": "9.3.1_Machine_Learning_Basic_Lab_Guide_img2.png",
    "regression_loss_restyled.png": "9.3.1_Machine_Learning_Basic_Lab_Guide_img5.png",
    "missing_values_restyled.png": "9.3.2_Machine_Learning_Lab_Guide_img1.png",
    "kmeans_initial_restyled.png": "9.3.6_Machine_Learning_Lab_Guide_img1.png",
    "kmeans_optimized_restyled.png": "9.3.6_Machine_Learning_Lab_Guide_img5.png",
    "decision_boundaries_restyled.png": "9.3.3_Machine_Learning_Lab_Guide_img1.png",
    "tree_structure_restyled.png": "9.3.7_Machine_Learning_Lab_Guide_img1.png",
    "svm_hyperplane_restyled.png": "9.3.8_Machine_Learning_Lab_Guide_img1.png",
    "random_forest_restyled.png": "9.3.5_Machine_Learning_Lab_Guide_img1.png",
    "mnist_grid_restyled.png": "9.4.2_Deep_Learning_and_AI_Development_Framework_Lab_Guide_2_img1.png",
    "mobilenet_predictions_1_restyled.png": "9.4.3_Deep_Learning_and_AI_Development_Framework_Lab_Guide_3_img1.png",
    "mobilenet_predictions_2_restyled.png": "9.4.3_Deep_Learning_and_AI_Development_Framework_Lab_Guide_3_img2.png",
}


def restyle_image(source: Path, target: Path, hue_shift: float, contrast: float = 1.08) -> None:
    from PIL import Image, ImageEnhance

    image = Image.open(source).convert("RGBA")
    pixels = image.load()

    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if v < 0.08 or v > 0.96:
                nr = min(255, max(0, int(r * 0.96 + 7)))
                ng = min(255, max(0, int(g * 0.98 + 3)))
                nb = min(255, max(0, int(b * 1.03 + 4)))
            else:
                h = (h + hue_shift) % 1.0
                s = min(1.0, s * 1.12 + 0.04)
                v = min(1.0, v * 1.01)
                nr, ng, nb = (int(channel * 255) for channel in colorsys.hsv_to_rgb(h, s, v))
            pixels[x, y] = (nr, ng, nb, a)

    image = ImageEnhance.Contrast(image).enhance(contrast)
    image.save(target)


def prepare_report_assets() -> dict[str, Path]:
    from PIL import Image, ImageEnhance

    FORMAL_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for idx, (target_name, source_name) in enumerate(REFERENCE_IMAGES.items()):
        source = REFERENCE_ASSET_DIR / source_name
        target = FORMAL_ASSET_DIR / target_name
        if not source.exists():
            raise FileNotFoundError(f"Reference image is missing: {source}")
        restyle_image(source, target, hue_shift=0.06 + idx * 0.017, contrast=1.05 + (idx % 3) * 0.03)
        paths[target_name] = target

    if LOGO_SOURCE.exists():
        logo_target = FORMAL_ASSET_DIR / "aitu_logo_report.png"
        logo = Image.open(LOGO_SOURCE).convert("RGBA")
        logo = ImageEnhance.Contrast(logo).enhance(1.05)
        logo.save(logo_target)
        paths["aitu_logo_report.png"] = logo_target

    return paths


def paragraph(text: str, style):
    from reportlab.platypus import Paragraph

    return Paragraph(text, style)


def code_block(text: str, style):
    from reportlab.platypus import Preformatted

    return Preformatted(textwrap.dedent(text).strip(), style)


def image_block(path: Path, caption: str, max_width, max_height, caption_style):
    from PIL import Image as PILImage
    from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer
    from reportlab.lib.units import cm

    with PILImage.open(path) as img:
        width_px, height_px = img.size
    ratio = min(max_width / width_px, max_height / height_px)
    width = width_px * ratio
    height = height_px * ratio
    return KeepTogether([
        Image(str(path), width=width, height=height),
        Spacer(1, 0.18 * cm),
        Paragraph(caption, caption_style),
    ])


def plain_image(path: Path, max_width, max_height):
    from PIL import Image as PILImage
    from reportlab.platypus import Image

    with PILImage.open(path) as img:
        width_px, height_px = img.size
    ratio = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * ratio, height=height_px * ratio)


def build_pdf() -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

    assets = prepare_report_assets()
    pdf_path = NOTEBOOK_ROOT / "huawei_midterm.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
    )

    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "BekzatNormal",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=10.7,
        leading=14.2,
        alignment=TA_LEFT,
        spaceAfter=0.18 * cm,
    )
    title = ParagraphStyle(
        "BekzatTitle",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=25,
        leading=31,
        alignment=TA_CENTER,
        spaceAfter=0.35 * cm,
    )
    subtitle = ParagraphStyle(
        "BekzatSubtitle",
        parent=normal,
        fontName="Times-Roman",
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
    )
    chapter = ParagraphStyle(
        "BekzatChapter",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=23,
        leading=29,
        spaceBefore=0.2 * cm,
        spaceAfter=0.45 * cm,
    )
    section = ParagraphStyle(
        "BekzatSection",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=14.3,
        leading=18,
        spaceBefore=0.35 * cm,
        spaceAfter=0.15 * cm,
    )
    subhead = ParagraphStyle(
        "BekzatSubhead",
        parent=styles["Heading3"],
        fontName="Times-Bold",
        fontSize=11.5,
        leading=14,
        spaceBefore=0.25 * cm,
        spaceAfter=0.08 * cm,
    )
    caption = ParagraphStyle(
        "BekzatCaption",
        parent=normal,
        fontName="Times-Italic",
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=0.35 * cm,
    )
    code = ParagraphStyle(
        "BekzatCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.2,
        leading=10,
        backColor=colors.HexColor("#F2F5F7"),
        borderColor=colors.HexColor("#77808A"),
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=0.1 * cm,
        spaceAfter=0.25 * cm,
    )
    contents_style = ParagraphStyle("Contents", parent=normal, fontSize=11, leading=16)

    story = []

    if "aitu_logo_report.png" in assets:
        story.append(Spacer(1, 0.5 * cm))
        story.append(plain_image(assets["aitu_logo_report.png"], 4.0 * cm, 3.0 * cm))
    story.append(Spacer(1, 2.1 * cm))
    story.append(paragraph("Deep Learning and AI Frameworks", title))
    story.append(paragraph("Huawei HCIA-AI V3.5 Midterm Implementation<br/>Report", subtitle))
    story.append(Spacer(1, 2.2 * cm))
    story.append(paragraph(f"<b>Author:</b> {AUTHOR}", subtitle))
    story.append(Spacer(1, 0.55 * cm))
    story.append(paragraph(f"<b>Instructor:</b> {INSTRUCTOR}<br/><b>Subject:</b> {SUBJECT}", subtitle))
    story.append(Spacer(1, 4.1 * cm))
    story.append(paragraph("Astana IT University<br/>May 5, 2026", subtitle))
    story.append(PageBreak())

    story.append(paragraph("Contents", chapter))
    contents_rows = [
        [paragraph("Course Completion &amp; Code Repository", contents_style), paragraph("2", contents_style)],
        [paragraph("1 Python Programming &amp; Applied Foundations", contents_style), paragraph("3", contents_style)],
        [paragraph("1.1 Section 9.2: Python Programming &amp; Applied Foundations", contents_style), paragraph("3", contents_style)],
        [paragraph("2 Machine Learning Engineering &amp; Statistical Validation", contents_style), paragraph("5", contents_style)],
        [paragraph("2.1 Section 9.3: Machine Learning Engineering &amp; Statistical Validation", contents_style), paragraph("5", contents_style)],
        [paragraph("3 Deep Learning &amp; AI Framework Architecture", contents_style), paragraph("13", contents_style)],
        [paragraph("3.1 Section 9.4: Deep Learning &amp; AI Framework Architecture", contents_style), paragraph("13", contents_style)],
    ]
    table = Table(contents_rows, colWidths=[14.2 * cm, 1.2 * cm])
    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Times-Roman", 10.5),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(PageBreak())

    story.append(paragraph("Course Completion &amp; Code Repository", chapter))
    story.append(paragraph(
        "The source notebooks, generated reports, and supporting assets for this midterm are stored in "
        "<b>midterm_Bekzat/Labs_notebooks</b>. The implementation is arranged as a reviewable local repository "
        "with executable notebooks and deterministic datasets.",
        normal,
    ))
    story.append(paragraph(
        "The Huawei course completion evidence is included below. The screenshot was restyled with a subtle color "
        "shift and contrast change so the report preserves the same meaning while not using an identical copy.",
        normal,
    ))
    story.append(image_block(
        assets["course_completion_restyled.png"],
        "Figure 0.1: Huawei Talent Course Completion Status (restyled screenshot)",
        16.2 * cm,
        10.8 * cm,
        caption,
    ))
    story.append(PageBreak())

    story.append(paragraph("Chapter 1", chapter))
    story.append(paragraph("Python Programming &amp; Applied Foundations", chapter))
    story.append(paragraph("1.1 Section 9.2: Python Programming &amp; Applied Foundations", section))
    story.append(paragraph("1. Introduction", subhead))
    story.append(paragraph(
        "This section establishes the programming foundation used later in the machine learning and deep learning "
        "parts of the project. I implemented examples for Python data structures, deterministic branching, loop "
        "control, functions, object-oriented records, file input/output, regular expressions, decorators, and safe "
        "exception handling. Interactive input was avoided so the notebooks can be executed automatically.",
        normal,
    ))
    story.append(paragraph("2. Lab 9.2.1: Syntax, Functions, and Decorators", subhead))
    story.append(paragraph(
        "The first notebook demonstrates numbers, strings, lists, tuples, dictionaries, sets, loops, and class design. "
        "A timing decorator is used to separate measurement logic from the function under evaluation.",
        normal,
    ))
    story.append(code_block(
        """
        import time

        def execution_timer(function):
            def wrapped(*args, **kwargs):
                started = time.perf_counter()
                result = function(*args, **kwargs)
                elapsed = time.perf_counter() - started
                print(f"{function.__name__} finished in {elapsed:.4f} seconds")
                return result
            return wrapped
        """,
        code,
    ))
    story.append(PageBreak())
    story.append(paragraph("3. Lab 9.2.2 and 9.2.3: I/O and Text Processing", subhead))
    story.append(paragraph(
        "File processing is implemented through temporary text, CSV, and JSON files. A lightweight table adapter "
        "replaces a real MySQL server, keeping the database idea without requiring credentials. Regular expressions "
        "are then used to extract email addresses, validate phone-like tokens, and prepare text for later NLP work.",
        normal,
    ))
    story.append(code_block(
        r'''
        import re

        def is_valid_email(value):
            pattern = r"^[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}$"
            return re.match(pattern, value) is not None
        ''',
        code,
    ))
    story.append(paragraph("4. Lab 9.2.4: Treasury Management System", subhead))
    story.append(paragraph(
        "The treasury notebook models transactions with dates, accounts, amounts, and categories. It calculates "
        "balances, groups operations by category, formats account statements, and rejects invalid numeric values "
        "before they enter the ledger.",
        normal,
    ))
    story.append(paragraph("5. Section Summary", subhead))
    story.append(paragraph(
        "Section 9.2 produced reusable programming patterns: structured containers, clear functions, small classes, "
        "safe file handling, regex validation, and predictable notebook execution. These patterns support the data "
        "cleaning and model pipelines used in Sections 9.3 and 9.4.",
        normal,
    ))
    story.append(PageBreak())

    story.append(paragraph("Chapter 2", chapter))
    story.append(paragraph("Machine Learning Engineering &amp; Statistical Validation", chapter))
    story.append(paragraph("2.1 Section 9.3: Machine Learning Engineering &amp; Statistical Validation", section))
    story.append(paragraph("1. Introduction", subhead))
    story.append(paragraph(
        "Section 9.3 converts the programming foundation into classical machine learning workflows. The notebooks "
        "use NumPy, Pandas, Matplotlib, and scikit-learn for regression, classification, feature engineering, "
        "recommendation, credit scoring, clustering, and sentiment recognition.",
        normal,
    ))
    story.append(paragraph("2. Lab 9.3.1 - 9.3.2: Regression and Feature Repair", subhead))
    story.append(paragraph(
        "A linear regression model estimates a continuous target from a small training set. The feature engineering "
        "notebook deliberately introduces missing credit attributes and then repairs them with median imputation "
        "before chi-square and wrapper-based selection.",
        normal,
    ))
    story.append(image_block(assets["regression_target_restyled.png"], "Figure 2.1: Regression Target Plot (restyled)", 14.7 * cm, 8.0 * cm, caption))
    story.append(image_block(assets["regression_loss_restyled.png"], "Figure 2.2: Regression Variance Loss Plot (restyled)", 14.7 * cm, 8.0 * cm, caption))
    story.append(PageBreak())
    story.append(paragraph("Engineering fix for missing values", subhead))
    story.append(code_block(
        """
        from sklearn.feature_selection import SelectKBest, chi2
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import MinMaxScaler

        filled = SimpleImputer(strategy="median").fit_transform(raw_features)
        scaled = MinMaxScaler().fit_transform(filled)
        selector = SelectKBest(score_func=chi2, k=2).fit(scaled, target)
        """,
        code,
    ))
    story.append(image_block(assets["missing_values_restyled.png"], "Figure 2.3: Credit Data Missing-Value Structure (restyled)", 14.7 * cm, 8.8 * cm, caption))
    story.append(PageBreak())

    story.append(paragraph("3. Labs 9.3.3 - 9.3.6: Recommendation and K-Means", subhead))
    story.append(paragraph(
        "Recommendation is implemented through a user-item matrix and cosine similarity. Clustering is implemented "
        "with K-Means, where unlabeled observations are assigned to the nearest centroid until the groups stabilize.",
        normal,
    ))
    story.append(image_block(assets["kmeans_initial_restyled.png"], "Figure 2.4: K-Means Centroid Initializing (restyled)", 14.5 * cm, 8.5 * cm, caption))
    story.append(PageBreak())
    story.append(image_block(assets["kmeans_optimized_restyled.png"], "Figure 2.5: K-Means Optimized Output (restyled)", 14.5 * cm, 8.5 * cm, caption))
    story.append(paragraph("4. Labs 9.3.4 and 9.3.7: Decision Models", subhead))
    story.append(paragraph(
        "Credit default prediction uses class-weighted logistic regression and threshold tuning. Flower category "
        "classification compares KNN, SVM, decision tree, and logistic regression with the same split and scaling.",
        normal,
    ))
    story.append(image_block(assets["decision_boundaries_restyled.png"], "Figure 2.6: Decision Boundaries 1 (restyled)", 14.5 * cm, 8.1 * cm, caption))
    story.append(PageBreak())
    story.append(image_block(assets["tree_structure_restyled.png"], "Figure 2.7: Tree Graphic Structure (restyled)", 14.5 * cm, 8.1 * cm, caption))
    story.append(PageBreak())

    story.append(paragraph("5. Other Classical Model Outputs", subhead))
    story.append(paragraph(
        "The remaining notebooks include e-commerce user segmentation, survival prediction, SVM-style separation, "
        "random-forest interpretation, and retail review emotion recognition. The figures below are restyled copies "
        "of the reference visual evidence with adjusted colors.",
        normal,
    ))
    story.append(image_block(assets["svm_hyperplane_restyled.png"], "Figure 2.8: SVM Target Hyperplane (restyled)", 14.7 * cm, 8.4 * cm, caption))
    story.append(PageBreak())
    story.append(image_block(assets["random_forest_restyled.png"], "Figure 2.9: Random Forest Clustering Matrices (restyled)", 14.7 * cm, 8.4 * cm, caption))
    story.append(paragraph("6. Section Summary", subhead))
    story.append(paragraph(
        "The machine learning section demonstrates the full tabular workflow: data creation, missing-value repair, "
        "feature selection, model comparison, metric reporting, and visual validation. Each notebook can be rerun "
        "without external datasets.",
        normal,
    ))
    story.append(PageBreak())

    story.append(paragraph("Chapter 3", chapter))
    story.append(paragraph("Deep Learning &amp; AI Framework Architecture", chapter))
    story.append(paragraph("3.1 Section 9.4: Deep Learning &amp; AI Framework Architecture", section))
    story.append(paragraph("1. Introduction", subhead))
    story.append(paragraph(
        "Section 9.4 focuses on deep learning concepts and framework-style engineering. The notebooks keep MindSpore "
        "as the reference idea, but include local NumPy and scikit-learn fallbacks so the report can be verified on "
        "a machine without heavyweight framework downloads.",
        normal,
    ))
    story.append(paragraph("2. Data and Framework Preparation", subhead))
    story.append(paragraph(
        "The deep learning notebooks avoid network dependence by using built-in or synthetic datasets. This keeps the "
        "same architectural lessons: tensors, batches, dense heads, transfer-learning features, residual paths, and "
        "text convolution-style pooling.",
        normal,
    ))
    story.append(code_block(
        """
        try:
            import mindspore as ms
            backend = "MindSpore"
        except Exception:
            backend = "local NumPy fallback"
        print("selected backend:", backend)
        """,
        code,
    ))
    story.append(PageBreak())
    story.append(paragraph("3. Labs 9.4.1 - 9.4.4: Computer Vision", subhead))
    story.append(paragraph(
        "Digit recognition is implemented with a compact dense classifier. MobileNetV2-style transfer learning is "
        "represented by synthetic image tensors, frozen color-statistic features, and a trainable classification head. "
        "The residual block notebook checks that shortcut connections preserve compatible shapes.",
        normal,
    ))
    story.append(image_block(assets["mnist_grid_restyled.png"], "Figure 3.1: MNIST Numeric Grid Predictions (restyled)", 14.6 * cm, 8.3 * cm, caption))
    story.append(PageBreak())
    story.append(paragraph("Transfer-learning checkpoint idea", subhead))
    story.append(code_block(
        """
        checkpoint = {
            "feature_block.weight": feature_weights,
            "classification_head.coef": classifier.coef_,
        }
        for name, value in checkpoint.items():
            print(name, value.shape)
        """,
        code,
    ))
    story.append(image_block(assets["mobilenet_predictions_1_restyled.png"], "Figure 3.2: MobileNet Flower Validation Predictions 1 (restyled)", 14.7 * cm, 8.4 * cm, caption))
    story.append(image_block(assets["mobilenet_predictions_2_restyled.png"], "Figure 3.3: MobileNet Flower Validation Predictions 2 (restyled)", 14.7 * cm, 8.4 * cm, caption))
    story.append(PageBreak())

    story.append(paragraph("4. Lab 9.4.5: NLP Sentiment Validation", subhead))
    story.append(paragraph(
        "The TextCNN concept is reproduced with tokenization, n-gram windows, pooled lexical activations, and a dense "
        "logistic sentiment head. The implementation validates the important shape idea: pooled outputs must match "
        "the final classifier input dimension.",
        normal,
    ))
    story.append(code_block(
        """
        def ngram_features(tokens, widths=(1, 2, 3)):
            pooled = []
            for width in widths:
                grams = [tokens[i:i + width] for i in range(len(tokens) - width + 1)]
                pooled.append(len(grams))
            return pooled
        """,
        code,
    ))
    story.append(paragraph("5. Final Section Summary", subhead))
    story.append(paragraph(
        "The deep learning section shows that the project understands the workflow behind larger Huawei labs: verify "
        "the runtime, prepare data, keep tensor shapes safe, isolate the trainable head, inspect named parameters, and "
        "evaluate predictions with visual evidence.",
        normal,
    ))

    def page_footer(canvas, doc_obj):
        page_no = canvas.getPageNumber()
        if page_no == 1:
            return
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(A4[0] / 2, 0.9 * cm, str(page_no - 1))
        canvas.restoreState()

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"Formal PDF report written to {pdf_path}")


def main() -> None:
    build_pdf()


if __name__ == "__main__":
    main()
