from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy  # type: ignore
from flask_login import (  # type: ignore
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv  # type: ignore
import pandas as pd
from datetime import datetime
from sqlalchemy import func  # type: ignore
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules  # type: ignore

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your-secret-key-here")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///retail.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# Household model
class Household(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hshd_num = db.Column(db.Integer, nullable=False)
    loyalty_flag = db.Column(db.String(1))
    age_range = db.Column(db.String(10))
    marital = db.Column(db.String(20))
    income_range = db.Column(db.String(20))
    homeowner = db.Column(db.String(20))
    hshd_composition = db.Column(db.String(50))
    hh_size = db.Column(db.String(10))
    children = db.Column(db.String(10))


# Transaction model
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    basket_num = db.Column(db.Integer, nullable=False)
    hshd_num = db.Column(db.Integer, nullable=False)
    purchase_date = db.Column(db.Date)
    product_num = db.Column(db.Integer, nullable=False)
    spend = db.Column(db.Float)
    units = db.Column(db.Integer)
    store_r = db.Column(db.String(20))
    week_num = db.Column(db.Integer)
    year = db.Column(db.Integer)


# Product model
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_num = db.Column(db.Integer, nullable=False)
    department = db.Column(db.String(50))
    commodity = db.Column(db.String(100))
    brand_ty = db.Column(db.String(20))
    natural_organic_flag = db.Column(db.String(1))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Data loading function
def load_data():
    try:
        print("\n=== Starting data load ===")

        # Load household data
        print("\nLoading household data...")
        household_df = pd.read_csv("data/400_households.csv", skipinitialspace=True)
        household_df.columns = household_df.columns.str.strip()
        print(f"Loaded {len(household_df)} households from CSV")
        print(f"Column names: {household_df.columns.tolist()}")

        for _, row in household_df.iterrows():
            try:
                household = Household(
                    hshd_num=int(row["HSHD_NUM"]),
                    loyalty_flag=row["L"],
                    age_range=(
                        row["AGE_RANGE"]
                        if str(row["AGE_RANGE"]).strip().lower() != "null"
                        else None
                    ),
                    marital=(
                        row["MARITAL"]
                        if str(row["MARITAL"]).strip().lower() != "null"
                        else None
                    ),
                    income_range=(
                        row["INCOME_RANGE"]
                        if str(row["INCOME_RANGE"]).strip().lower() != "null"
                        else None
                    ),
                    homeowner=(
                        row["HOMEOWNER"]
                        if str(row["HOMEOWNER"]).strip().lower() != "null"
                        else None
                    ),
                    hshd_composition=(
                        row["HSHD_COMPOSITION"]
                        if str(row["HSHD_COMPOSITION"]).strip().lower() != "null"
                        else None
                    ),
                    hh_size=(
                        str(row["HH_SIZE"])
                        if str(row["HH_SIZE"]).strip().lower() != "null"
                        else None
                    ),
                    children=(
                        row["CHILDREN"]
                        if str(row["CHILDREN"]).strip().lower() != "null"
                        else None
                    ),
                )
                db.session.add(household)
            except Exception as e:
                print(f"Error processing row: {row.to_dict()}")
                print(f"Error details: {str(e)}")
                continue

        db.session.commit()
        print(f"Successfully loaded {Household.query.count()} households into database")

        # Load transaction data (limited to 25,000 rows)
        print("\nLoading transaction data...")
        transaction_df = pd.read_csv(
            "data/400_transactions.csv", skipinitialspace=True, nrows=25000
        )
        transaction_df.columns = transaction_df.columns.str.strip()
        print(f"Loaded {len(transaction_df)} transactions from CSV")
        print(f"Column names: {transaction_df.columns.tolist()}")
        print(f"Sample data: {transaction_df.head().to_dict()}")

        transaction_count = 0
        for _, row in transaction_df.iterrows():
            try:
                # Parse the date string into a Python date object
                purchase_date = datetime.strptime(row["PURCHASE_"], "%d-%b-%y").date()

                transaction = Transaction(
                    basket_num=row["BASKET_NUM"],
                    hshd_num=row["HSHD_NUM"],
                    purchase_date=purchase_date,
                    product_num=row["PRODUCT_NUM"],
                    spend=row["SPEND"],
                    units=row["UNITS"],
                    store_r=row["STORE_R"],
                    week_num=row["WEEK_NUM"],
                    year=row["YEAR"],
                )
                db.session.add(transaction)
                transaction_count += 1

                # Commit every 1000 transactions to avoid memory issues
                if transaction_count % 1000 == 0:
                    db.session.commit()
            except Exception as e:
                print(f"Error processing transaction row: {row.to_dict()}")
                print(f"Error details: {str(e)}")
                continue

        db.session.commit()
        print(
            f"Successfully loaded {Transaction.query.count()} transactions into database"
        )

        # Load product data
        print("\nLoading product data...")
        product_df = pd.read_csv("data/400_products.csv", skipinitialspace=True)
        product_df.columns = product_df.columns.str.strip()
        print(f"Loaded {len(product_df)} products from CSV")
        print(f"Column names: {product_df.columns.tolist()}")
        print(f"Sample data: {product_df.head().to_dict()}")

        product_count = 0
        for _, row in product_df.iterrows():
            try:
                product = Product(
                    product_num=row["PRODUCT_NUM"],
                    department=row["DEPARTMENT"],
                    commodity=row["COMMODITY"],
                    brand_ty=row["BRAND_TY"],
                    natural_organic_flag=row["NATURAL_ORGANIC_FLAG"],
                )
                db.session.add(product)
                product_count += 1

                # Commit every 1000 products to avoid memory issues
                if product_count % 1000 == 0:
                    db.session.commit()
            except Exception as e:
                print(f"Error processing product row: {row.to_dict()}")
                print(f"Error details: {str(e)}")
                continue

        db.session.commit()
        print(f"Successfully loaded {Product.query.count()} products into database")

        print("\n=== Data load completed ===")
        return True
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        db.session.rollback()
        return False


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid username or password")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if User.query.filter_by(username=username).first():
            flash("Username already exists")
            return redirect(url_for("signup"))

        if User.query.filter_by(email=email).first():
            flash("Email already exists")
            return redirect(url_for("signup"))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# Dashboard Routes
@app.route("/dashboard")
@login_required
def dashboard():
    # Get quick stats
    total_households = Household.query.count()
    total_transactions = Transaction.query.count()
    total_products = Product.query.count()
    total_revenue = db.session.query(db.func.sum(Transaction.spend)).scalar() or 0

    # Check if data exists
    if total_households == 0:
        print("No data found in database. Attempting to load data...")
        load_data()
        # Refresh stats after loading
        total_households = Household.query.count()
        total_transactions = Transaction.query.count()
        total_products = Product.query.count()
        total_revenue = db.session.query(db.func.sum(Transaction.spend)).scalar() or 0

    return render_template(
        "dashboard.html",
        total_households=total_households,
        total_transactions=total_transactions,
        total_products=total_products,
        total_revenue=total_revenue,
    )


@app.route("/sample_data")
@login_required
def sample_data():
    # Get data for household #10
    household = Household.query.filter_by(hshd_num=10).first()

    if household:
        # Get transactions sorted by basket_num, purchase_date, product_num
        transactions = (
            Transaction.query.filter_by(hshd_num=10)
            .order_by(
                Transaction.basket_num,
                Transaction.purchase_date,
                Transaction.product_num,
            )
            .all()
        )

        if transactions:
            # Get product information for those transactions
            product_nums = [t.product_num for t in transactions]
            products_dict = {}

            products = Product.query.filter(Product.product_num.in_(product_nums)).all()

            # Create a dictionary of products for quick lookup
            for product in products:
                products_dict[product.product_num] = product

            # Create combined data for display
            combined_data = []
            for transaction in transactions:
                product = products_dict.get(transaction.product_num)
                if product:
                    combined_data.append(
                        {
                            "hshd_num": transaction.hshd_num,
                            "basket_num": transaction.basket_num,
                            "purchase_date": transaction.purchase_date,
                            "product_num": transaction.product_num,
                            "department": product.department,
                            "commodity": product.commodity,
                            "spend": transaction.spend,
                            "units": transaction.units,
                            "store_r": transaction.store_r,
                            "week_num": transaction.week_num,
                            "year": transaction.year,
                            "brand_ty": product.brand_ty,
                            "natural_organic_flag": product.natural_organic_flag,
                        }
                    )

            # Limit to first 10 rows for sample display
            combined_data = combined_data[:10]
        else:
            combined_data = []
    else:
        transactions = []
        products = []
        combined_data = []

    return render_template(
        "sample_data.html", household=household, combined_data=combined_data
    )


@app.route("/search_data")
@login_required
def search_data():
    hshd_num = request.args.get("hshd_num")
    household = None
    combined_data = []

    if hshd_num:
        try:
            hshd_num = int(hshd_num)
            household = Household.query.filter_by(hshd_num=hshd_num).first()
            if household:
                # Get transactions sorted by basket_num, purchase_date, product_num
                transactions = (
                    Transaction.query.filter_by(hshd_num=hshd_num)
                    .order_by(
                        Transaction.basket_num,
                        Transaction.purchase_date,
                        Transaction.product_num,
                    )
                    .all()
                )

                if transactions:
                    # Get product information for those transactions
                    product_nums = [t.product_num for t in transactions]
                    products_dict = {}

                    products = (
                        Product.query.filter(Product.product_num.in_(product_nums))
                        .order_by(Product.department, Product.commodity)
                        .all()
                    )

                    # Create a dictionary of products for quick lookup
                    for product in products:
                        products_dict[product.product_num] = product

                    # Create combined data for display
                    for transaction in transactions:
                        product = products_dict.get(transaction.product_num)
                        if product:
                            combined_data.append(
                                {
                                    "hshd_num": transaction.hshd_num,
                                    "basket_num": transaction.basket_num,
                                    "purchase_date": transaction.purchase_date,
                                    "product_num": transaction.product_num,
                                    "department": product.department,
                                    "commodity": product.commodity,
                                    "spend": transaction.spend,
                                    "units": transaction.units,
                                    "store_r": transaction.store_r,
                                    "week_num": transaction.week_num,
                                    "year": transaction.year,
                                    "brand_ty": product.brand_ty,
                                    "natural_organic_flag": product.natural_organic_flag,
                                }
                            )
        except ValueError:
            flash("Invalid household number")
            return redirect(url_for("search_data"))

    return render_template(
        "search_data.html",
        household=household,
        combined_data=combined_data,
        search_query=hshd_num,
    )


@app.route("/upload_data")
@login_required
def upload_data():
    return render_template("upload_data.html")


@app.route("/upload_data_process", methods=["POST"])
@login_required
def upload_data_process():
    try:
        results = {
            "success": True,
            "message": "Data uploaded successfully",
            "details": {},
        }

        # Process households file if provided
        if "households" in request.files:
            file = request.files["households"]
            if file.filename:
                # Verify file name
                if file.filename != "400_households.csv":
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": "Household file must be named 400_households.csv",
                            }
                        ),
                        400,
                    )

                # Save the file to data directory, replacing the existing file
                file_path = os.path.join("data", "400_households.csv")
                file.save(file_path)

                # Update database
                result = update_households_data(file_path)
                results["details"]["households"] = result

        # Process transactions file if provided
        if "transactions" in request.files:
            file = request.files["transactions"]
            if file.filename:
                # Verify file name
                if file.filename != "400_transactions.csv":
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": "Transactions file must be named 400_transactions.csv",
                            }
                        ),
                        400,
                    )

                # Save the file to data directory, replacing the existing file
                file_path = os.path.join("data", "400_transactions.csv")
                file.save(file_path)

                # Update database
                result = update_transactions_data(file_path)
                results["details"]["transactions"] = result

        # Process products file if provided
        if "products" in request.files:
            file = request.files["products"]
            if file.filename:
                # Verify file name
                if file.filename != "400_products.csv":
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": "Products file must be named 400_products.csv",
                            }
                        ),
                        400,
                    )

                # Save the file to data directory, replacing the existing file
                file_path = os.path.join("data", "400_products.csv")
                file.save(file_path)

                # Update database
                result = update_products_data(file_path)
                results["details"]["products"] = result

        return jsonify(results)

    except Exception as e:
        print(f"Error processing uploaded data: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


def update_households_data(file_path):
    """Update household data in the database from the uploaded file."""
    try:
        # Read the CSV file
        household_df = pd.read_csv(file_path, skipinitialspace=True)
        household_df.columns = household_df.columns.str.strip()

        # Clear existing data
        Household.query.delete()
        db.session.commit()

        # Process rows
        processed_count = 0
        for _, row in household_df.iterrows():
            try:
                household = Household(
                    hshd_num=int(row["HSHD_NUM"]),
                    loyalty_flag=row["L"],
                    age_range=(
                        row["AGE_RANGE"]
                        if str(row["AGE_RANGE"]).strip().lower() != "null"
                        else None
                    ),
                    marital=(
                        row["MARITAL"]
                        if str(row["MARITAL"]).strip().lower() != "null"
                        else None
                    ),
                    income_range=(
                        row["INCOME_RANGE"]
                        if str(row["INCOME_RANGE"]).strip().lower() != "null"
                        else None
                    ),
                    homeowner=(
                        row["HOMEOWNER"]
                        if str(row["HOMEOWNER"]).strip().lower() != "null"
                        else None
                    ),
                    hshd_composition=(
                        row["HSHD_COMPOSITION"]
                        if str(row["HSHD_COMPOSITION"]).strip().lower() != "null"
                        else None
                    ),
                    hh_size=(
                        str(row["HH_SIZE"])
                        if str(row["HH_SIZE"]).strip().lower() != "null"
                        else None
                    ),
                    children=(
                        row["CHILDREN"]
                        if str(row["CHILDREN"]).strip().lower() != "null"
                        else None
                    ),
                )
                db.session.add(household)
                processed_count += 1

                # Commit every 100 records
                if processed_count % 100 == 0:
                    db.session.commit()

            except Exception as e:
                print(f"Error processing household row: {row.to_dict()}")
                print(f"Error details: {str(e)}")
                continue

        # Commit remaining records
        db.session.commit()
        return f"{processed_count} households processed"
    except Exception as e:
        db.session.rollback()
        raise Exception(f"Error updating household data: {str(e)}")


def update_transactions_data(file_path):
    """Update transaction data in the database from the uploaded file."""
    try:
        # Read the CSV file
        transaction_df = pd.read_csv(file_path, skipinitialspace=True)
        transaction_df.columns = transaction_df.columns.str.strip()

        # Clear existing data
        Transaction.query.delete()
        db.session.commit()

        # Process rows (limit to 25,000 for performance)
        processed_count = 0
        for _, row in transaction_df.iterrows():
            if processed_count >= 25000:
                break

            try:
                # Parse the date string into a Python date object
                purchase_date = datetime.strptime(row["PURCHASE_"], "%d-%b-%y").date()

                transaction = Transaction(
                    basket_num=row["BASKET_NUM"],
                    hshd_num=row["HSHD_NUM"],
                    purchase_date=purchase_date,
                    product_num=row["PRODUCT_NUM"],
                    spend=row["SPEND"],
                    units=row["UNITS"],
                    store_r=row["STORE_R"],
                    week_num=row["WEEK_NUM"],
                    year=row["YEAR"],
                )
                db.session.add(transaction)
                processed_count += 1

                # Commit every 1000 transactions to avoid memory issues
                if processed_count % 1000 == 0:
                    db.session.commit()

            except Exception as e:
                print(f"Error processing transaction row: {row.to_dict()}")
                print(f"Error details: {str(e)}")
                continue

        # Commit remaining records
        db.session.commit()
        return f"{processed_count} transactions processed"
    except Exception as e:
        db.session.rollback()
        raise Exception(f"Error updating transaction data: {str(e)}")


def update_products_data(file_path):
    """Update product data in the database from the uploaded file."""
    try:
        # Read the CSV file
        product_df = pd.read_csv(file_path, skipinitialspace=True)
        product_df.columns = product_df.columns.str.strip()

        # Clear existing data
        Product.query.delete()
        db.session.commit()

        # Process rows
        processed_count = 0
        for _, row in product_df.iterrows():
            try:
                product = Product(
                    product_num=row["PRODUCT_NUM"],
                    department=row["DEPARTMENT"],
                    commodity=row["COMMODITY"],
                    brand_ty=row["BRAND_TY"],
                    natural_organic_flag=row["NATURAL_ORGANIC_FLAG"],
                )
                db.session.add(product)
                processed_count += 1

                # Commit every 100 products to avoid memory issues
                if processed_count % 100 == 0:
                    db.session.commit()

            except Exception as e:
                print(f"Error processing product row: {row.to_dict()}")
                print(f"Error details: {str(e)}")
                continue

        # Commit remaining records
        db.session.commit()
        return f"{processed_count} products processed"
    except Exception as e:
        db.session.rollback()
        raise Exception(f"Error updating product data: {str(e)}")


@app.route("/customer_analytics")
@login_required
def customer_analytics():
    return render_template("customer_analytics.html")


@app.route("/product_analytics")
@login_required
def product_analytics():
    return render_template("product_analytics.html")


@app.route("/sales_analytics")
@login_required
def sales_analytics():
    return render_template("sales_analytics.html")


@app.route("/basket_analysis")
@login_required
def basket_analysis():
    return render_template("basket_analysis.html")


@app.route("/churn_prediction")
@login_required
def churn_prediction():
    return render_template("churn_prediction.html")


# Add a route to check data status
@app.route("/check_data")
@login_required
def check_data():
    stats = {
        "households": Household.query.count(),
        "transactions": Transaction.query.count(),
        "products": Product.query.count(),
        "revenue": db.session.query(db.func.sum(Transaction.spend)).scalar() or 0,
    }
    return jsonify(stats)


@app.route("/customer_analytics_data")
@login_required
def customer_analytics_data():
    try:
        # Customer Segmentation Data
        income_segmentation = (
            db.session.query(Household.income_range, func.count(Household.id))
            .group_by(Household.income_range)
            .all()
        )

        # Household Size Segmentation - Updated query
        household_size_segmentation = (
            db.session.query(Household.hh_size, func.count(Household.id))
            .group_by(Household.hh_size)
            .all()
        )

        age_segmentation = (
            db.session.query(Household.age_range, func.count(Household.id))
            .group_by(Household.age_range)
            .all()
        )

        loyalty_segmentation = (
            db.session.query(Household.loyalty_flag, func.count(Household.id))
            .group_by(Household.loyalty_flag)
            .all()
        )

        # Purchase Behavior Data
        top_categories = (
            db.session.query(Product.department, func.count(Transaction.id))
            .join(Transaction, Product.product_num == Transaction.product_num)
            .group_by(Product.department)
            .order_by(func.count(Transaction.id).desc())
            .limit(10)
            .all()
        )

        # Prepare data for charts
        data = {
            "incomeSegmentation": {
                "type": "pie",
                "data": {
                    "labels": [
                        str(income[0]) if income[0] is not None else "Unknown"
                        for income in income_segmentation
                    ],
                    "datasets": [
                        {
                            "data": [income[1] for income in income_segmentation],
                            "backgroundColor": [
                                "#FF6384",
                                "#36A2EB",
                                "#FFCE56",
                                "#4BC0C0",
                                "#9966FF",
                            ],
                        }
                    ],
                },
            },
            "householdSizeSegmentation": {
                "type": "bar",
                "data": {
                    "labels": [
                        (
                            f"{size[0].strip()} people"
                            if size[0] is not None
                            else "Unknown"
                        )
                        for size in household_size_segmentation
                    ],
                    "datasets": [
                        {
                            "label": "Number of Households",
                            "data": [size[1] for size in household_size_segmentation],
                            "backgroundColor": "#36A2EB",
                        }
                    ],
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {"display": True, "text": "Number of Households"},
                        },
                        "x": {"title": {"display": True, "text": "Household Size"}},
                    }
                },
            },
            "ageSegmentation": {
                "type": "pie",
                "data": {
                    "labels": [
                        str(age[0]) if age[0] is not None else "Unknown"
                        for age in age_segmentation
                    ],
                    "datasets": [
                        {
                            "data": [age[1] for age in age_segmentation],
                            "backgroundColor": [
                                "#FF6384",
                                "#36A2EB",
                                "#FFCE56",
                                "#4BC0C0",
                            ],
                        }
                    ],
                },
            },
            "loyaltySegmentation": {
                "type": "pie",
                "data": {
                    "labels": [
                        "Loyal" if flag[0] == "Y" else "Non-Loyal"
                        for flag in loyalty_segmentation
                    ],
                    "datasets": [
                        {
                            "data": [flag[1] for flag in loyalty_segmentation],
                            "backgroundColor": ["#4BC0C0", "#FF6384"],
                        }
                    ],
                },
            },
            "topCategories": {
                "type": "bar",
                "data": {
                    "labels": [
                        str(cat[0]) if cat[0] is not None else "Unknown"
                        for cat in top_categories
                    ],
                    "datasets": [
                        {
                            "label": "Number of Purchases",
                            "data": [cat[1] for cat in top_categories],
                            "backgroundColor": "#36A2EB",
                        }
                    ],
                },
            },
        }

        return jsonify(data)
    except Exception as e:
        print(f"Error generating analytics data: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/product_analytics_data")
@login_required
def product_analytics_data():
    try:
        # Product Performance Data
        top_products = (
            db.session.query(
                Product.product_num,
                func.count(Transaction.id).label("sales_count"),
                func.sum(Transaction.spend).label("total_revenue"),
            )
            .join(Transaction, Product.product_num == Transaction.product_num)
            .group_by(Product.product_num)
            .order_by(func.count(Transaction.id).desc())
            .limit(10)
            .all()
        )

        # Category Analysis Data
        sales_by_department = (
            db.session.query(
                Product.department,
                func.count(Transaction.id).label("sales_count"),
                func.sum(Transaction.spend).label("total_revenue"),
            )
            .join(Transaction, Product.product_num == Transaction.product_num)
            .group_by(Product.department)
            .order_by(func.count(Transaction.id).desc())
            .all()
        )

        top_commodities = (
            db.session.query(
                Product.commodity, func.count(Transaction.id).label("sales_count")
            )
            .join(Transaction, Product.product_num == Transaction.product_num)
            .group_by(Product.commodity)
            .order_by(func.count(Transaction.id).desc())
            .limit(10)
            .all()
        )

        # Brand Analysis Data
        brand_type_distribution = (
            db.session.query(
                Product.brand_ty, func.count(Transaction.id).label("sales_count")
            )
            .join(Transaction, Product.product_num == Transaction.product_num)
            .group_by(Product.brand_ty)
            .all()
        )

        organic_distribution = (
            db.session.query(
                Product.natural_organic_flag,
                func.count(Transaction.id).label("sales_count"),
            )
            .join(Transaction, Product.product_num == Transaction.product_num)
            .group_by(Product.natural_organic_flag)
            .all()
        )

        # Sales Trend Data
        sales_trend = (
            db.session.query(
                Transaction.purchase_date,
                func.count(Transaction.id).label("sales_count"),
                func.sum(Transaction.spend).label("total_revenue"),
            )
            .group_by(Transaction.purchase_date)
            .order_by(Transaction.purchase_date)
            .all()
        )

        # Prepare data for charts
        data = {
            "topProducts": {
                "type": "bar",
                "data": {
                    "labels": [str(product[0]) for product in top_products],
                    "datasets": [
                        {
                            "label": "Number of Sales",
                            "data": [product[1] for product in top_products],
                            "backgroundColor": "#36A2EB",
                        }
                    ],
                },
            },
            "revenueByProduct": {
                "type": "bar",
                "data": {
                    "labels": [str(product[0]) for product in top_products],
                    "datasets": [
                        {
                            "label": "Total Revenue",
                            "data": [float(product[2]) for product in top_products],
                            "backgroundColor": "#FFCE56",
                        }
                    ],
                },
            },
            "salesTrend": {
                "type": "line",
                "data": {
                    "labels": [date[0].strftime("%Y-%m-%d") for date in sales_trend],
                    "datasets": [
                        {
                            "label": "Number of Sales",
                            "data": [date[1] for date in sales_trend],
                            "borderColor": "#FF6384",
                            "fill": False,
                        }
                    ],
                },
            },
            "salesByDepartment": {
                "type": "pie",
                "data": {
                    "labels": [
                        str(dept[0]) if dept[0] is not None else "Unknown"
                        for dept in sales_by_department
                    ],
                    "datasets": [
                        {
                            "data": [dept[1] for dept in sales_by_department],
                            "backgroundColor": [
                                "#FF6384",
                                "#36A2EB",
                                "#FFCE56",
                                "#4BC0C0",
                                "#9966FF",
                            ],
                        }
                    ],
                },
            },
            "topCommodities": {
                "type": "bar",
                "data": {
                    "labels": [
                        str(comm[0]) if comm[0] is not None else "Unknown"
                        for comm in top_commodities
                    ],
                    "datasets": [
                        {
                            "label": "Number of Sales",
                            "data": [comm[1] for comm in top_commodities],
                            "backgroundColor": "#4BC0C0",
                        }
                    ],
                },
            },
            "brandTypeDistribution": {
                "type": "pie",
                "data": {
                    "labels": [
                        str(brand[0]) if brand[0] is not None else "Unknown"
                        for brand in brand_type_distribution
                    ],
                    "datasets": [
                        {
                            "data": [brand[1] for brand in brand_type_distribution],
                            "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56"],
                        }
                    ],
                },
            },
            "organicDistribution": {
                "type": "pie",
                "data": {
                    "labels": [
                        "Organic" if org[0] == "Y" else "Non-Organic"
                        for org in organic_distribution
                    ],
                    "datasets": [
                        {
                            "data": [org[1] for org in organic_distribution],
                            "backgroundColor": ["#4BC0C0", "#FF6384"],
                        }
                    ],
                },
            },
        }

        return jsonify(data)
    except Exception as e:
        print(f"Error generating product analytics data: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/sales_analytics_data")
@login_required
def sales_analytics_data():
    try:
        # Sales Trends Data
        daily_sales = (
            db.session.query(
                Transaction.purchase_date,
                func.count(Transaction.id).label("sales_count"),
                func.sum(Transaction.spend).label("total_revenue"),
            )
            .group_by(Transaction.purchase_date)
            .order_by(Transaction.purchase_date)
            .all()
        )

        weekly_sales = (
            db.session.query(
                Transaction.week_num,
                func.count(Transaction.id).label("sales_count"),
                func.sum(Transaction.spend).label("total_revenue"),
            )
            .group_by(Transaction.week_num)
            .order_by(Transaction.week_num)
            .all()
        )

        monthly_sales = (
            db.session.query(
                func.extract("month", Transaction.purchase_date).label("month"),
                func.count(Transaction.id).label("sales_count"),
                func.sum(Transaction.spend).label("total_revenue"),
            )
            .group_by("month")
            .order_by("month")
            .all()
        )

        # Revenue Analysis Data
        revenue_by_department = (
            db.session.query(
                Product.department, func.sum(Transaction.spend).label("total_revenue")
            )
            .join(Transaction, Product.product_num == Transaction.product_num)
            .group_by(Product.department)
            .order_by(func.sum(Transaction.spend).desc())
            .all()
        )

        avg_transaction_value = db.session.query(
            func.avg(Transaction.spend).label("avg_value")
        ).all()

        # Basket Analysis Data
        basket_size = (
            db.session.query(
                Transaction.basket_num,
                func.count(Transaction.id).label("item_count"),
                func.sum(Transaction.spend).label("total_value"),
            )
            .group_by(Transaction.basket_num)
            .all()
        )

        # Prepare data for charts
        data = {
            "dailySales": {
                "type": "line",
                "data": {
                    "labels": [date[0].strftime("%Y-%m-%d") for date in daily_sales],
                    "datasets": [
                        {
                            "label": "Number of Sales",
                            "data": [date[1] for date in daily_sales],
                            "borderColor": "#FF6384",
                            "fill": False,
                        }
                    ],
                },
            },
            "weeklySales": {
                "type": "bar",
                "data": {
                    "labels": [f"Week {week[0]}" for week in weekly_sales],
                    "datasets": [
                        {
                            "label": "Number of Sales",
                            "data": [week[1] for week in weekly_sales],
                            "backgroundColor": "#36A2EB",
                        }
                    ],
                },
            },
            "monthlySales": {
                "type": "line",
                "data": {
                    "labels": [f"Month {month[0]}" for month in monthly_sales],
                    "datasets": [
                        {
                            "label": "Number of Sales",
                            "data": [month[1] for month in monthly_sales],
                            "borderColor": "#FFCE56",
                            "fill": False,
                        }
                    ],
                },
            },
            "revenueByDepartment": {
                "type": "pie",
                "data": {
                    "labels": [
                        str(dept[0]) if dept[0] is not None else "Unknown"
                        for dept in revenue_by_department
                    ],
                    "datasets": [
                        {
                            "data": [float(dept[1]) for dept in revenue_by_department],
                            "backgroundColor": [
                                "#FF6384",
                                "#36A2EB",
                                "#FFCE56",
                                "#4BC0C0",
                                "#9966FF",
                            ],
                        }
                    ],
                },
            },
            "avgTransactionValue": {
                "type": "bar",
                "data": {
                    "labels": ["Average Transaction Value"],
                    "datasets": [
                        {
                            "data": [
                                float(avg[0]) if avg[0] is not None else 0
                                for avg in avg_transaction_value
                            ],
                            "backgroundColor": "#4BC0C0",
                        }
                    ],
                },
            },
            "avgBasketSize": {
                "type": "bar",
                "data": {
                    "labels": ["Average Items per Basket"],
                    "datasets": [
                        {
                            "data": [
                                sum(basket[1] for basket in basket_size)
                                / len(basket_size)
                            ],
                            "backgroundColor": "#FF6384",
                        }
                    ],
                },
            },
            "basketValueDistribution": {
                "type": "line",
                "data": {
                    "labels": [str(i) for i in range(len(basket_size))],
                    "datasets": [
                        {
                            "label": "Basket Value",
                            "data": sorted(
                                [float(basket[2]) for basket in basket_size]
                            ),
                            "borderColor": "#36A2EB",
                            "fill": False,
                        }
                    ],
                },
            },
        }

        return jsonify(data)
    except Exception as e:
        print(f"Error generating sales analytics data: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/basket_analysis_data")
@login_required
def basket_analysis_data():
    try:
        # 1. Get transaction data with product information
        transactions_query = (
            db.session.query(
                Transaction.basket_num, Transaction.product_num, Product.commodity
            )
            .join(Product, Transaction.product_num == Product.product_num)
            .all()
        )

        # Create dataframe
        df = pd.DataFrame(
            transactions_query, columns=["basket_num", "product_num", "commodity"]
        )
        print(f"Total transactions with product info: {len(df)}")

        # 2. Check if we have sufficient data
        if len(df) == 0:
            print("No transaction data found.")
            return jsonify({"rules": []})

        # 3. Group by basket to get products in each basket
        basket_products = (
            df.groupby("basket_num")["commodity"].apply(list).reset_index()
        )
        print(f"Total baskets: {len(basket_products)}")

        if len(basket_products) < 2:
            print("Not enough baskets for analysis.")
            return jsonify({"rules": [], "top_products": [], "co_occurrence": []})

        # 4. Create a product co-occurrence matrix
        # Get all unique products
        all_products = sorted(df["commodity"].unique())
        co_occurrence_matrix = pd.DataFrame(0, index=all_products, columns=all_products)

        # Fill the co-occurrence matrix
        for _, row in basket_products.iterrows():
            products = row["commodity"]
            # Count co-occurrences (including self-occurrences)
            for i, prod1 in enumerate(products):
                for prod2 in products:
                    if (
                        prod1 in co_occurrence_matrix.index
                        and prod2 in co_occurrence_matrix.columns
                    ):
                        co_occurrence_matrix.at[prod1, prod2] += 1

        # 5. Get top product combinations (excluding self-pairs)
        pairs = []
        for prod1 in co_occurrence_matrix.index:
            for prod2 in co_occurrence_matrix.columns:
                if prod1 != prod2:  # Exclude self-pairs
                    count = co_occurrence_matrix.at[prod1, prod2]
                    if count > 0:
                        # Calculate confidence: P(prod2|prod1)
                        confidence = count / co_occurrence_matrix.at[prod1, prod1]
                        # Calculate lift: P(prod1,prod2) / (P(prod1) * P(prod2))
                        lift = (count / len(basket_products)) / (
                            (
                                co_occurrence_matrix.at[prod1, prod1]
                                / len(basket_products)
                            )
                            * (
                                co_occurrence_matrix.at[prod2, prod2]
                                / len(basket_products)
                            )
                        )
                        support = count / len(basket_products)
                        pairs.append((prod1, prod2, count, support, confidence, lift))

        # Sort by count (most frequent pairs first)
        pairs.sort(key=lambda x: x[2], reverse=True)

        # 6. Get top products by frequency (for display)
        product_counts = df["commodity"].value_counts().reset_index()
        product_counts.columns = ["commodity", "count"]
        top_products = product_counts.head(10).to_dict("records")

        # 7. Create data suitable for visualization
        # Top product combinations for rules table
        results = []
        for prod1, prod2, count, support, confidence, lift in pairs[
            :20
        ]:  # Top 20 pairs
            results.append(
                {
                    "antecedents": [prod1],
                    "consequents": [prod2],
                    "support": round(support, 3),
                    "confidence": round(confidence, 3),
                    "lift": round(lift, 3),
                }
            )

        # 8. Create co-occurrence data for visualization
        top_products_list = [item["commodity"] for item in top_products][
            :10
        ]  # Use top 10 products
        co_occurrence_subset = co_occurrence_matrix.loc[
            top_products_list, top_products_list
        ]
        co_occurrence_data = {
            "products": top_products_list,
            "matrix": co_occurrence_subset.values.tolist(),
        }

        # 9. Return all data
        return jsonify(
            {
                "rules": results,
                "top_products": top_products,
                "co_occurrence": co_occurrence_data,
            }
        )
    except Exception as e:
        print(f"Error in basket analysis: {e}")
        import traceback

        traceback.print_exc()
        return (
            jsonify(
                {"error": str(e), "rules": [], "top_products": [], "co_occurrence": []}
            ),
            500,
        )


@app.route("/churn_prediction_data")
@login_required
def churn_prediction_data():
    try:
        # 1. Get transaction data with dates
        transaction_data = db.session.query(
            Transaction.hshd_num,
            Transaction.purchase_date,
            Transaction.spend,
            Transaction.week_num,
            Transaction.year,
        ).all()

        # Convert to pandas DataFrame for analysis
        df = pd.DataFrame(
            transaction_data,
            columns=["household_id", "purchase_date", "spend", "week_num", "year"],
        )

        # 2. Group by household and analyze their purchase patterns
        # Get all unique household IDs
        all_households = (
            df["household_id"].unique().tolist()
        )  # Convert numpy array to list

        # Get the min and max dates to establish the full time range
        min_date = df["purchase_date"].min()
        max_date = df["purchase_date"].max()
        total_weeks = int(len(df["week_num"].unique()))  # Convert to int

        # Assuming the data spans at least 4 weeks
        recency_threshold = (
            4  # Consider a household at risk if not active in the last 4 weeks
        )

        # Calculate metrics for each household
        household_metrics = []

        for hshd in all_households:
            # Get this household's transactions
            hshd_df = df[df["household_id"] == hshd]

            # Calculate recency (weeks since last purchase)
            max_week = int(hshd_df["week_num"].max())  # Convert to int
            overall_max_week = int(df["week_num"].max())  # Convert to int
            recency = overall_max_week - max_week

            # Calculate frequency (average weeks between purchases)
            unique_weeks = int(hshd_df["week_num"].nunique())  # Convert to int
            purchase_span = int(
                hshd_df["week_num"].max() - hshd_df["week_num"].min() + 1
            )  # Convert to int
            frequency = float(
                purchase_span / unique_weeks if unique_weeks > 1 else purchase_span
            )  # Convert to float

            # Calculate monetary value (average spend per transaction)
            monetary = float(hshd_df["spend"].mean())  # Convert to float

            # Calculate trend (are they spending more or less over time?)
            # Group by week and calculate total spend per week
            weekly_spend = hshd_df.groupby("week_num")["spend"].sum().reset_index()

            # If we have at least 2 weeks of data, calculate trend
            if len(weekly_spend) >= 2:
                # Simple linear regression
                X = weekly_spend["week_num"].values.reshape(-1, 1)
                y = weekly_spend["spend"].values

                # Calculate slope using numpy's polyfit
                slope = float(
                    np.polyfit(weekly_spend["week_num"], weekly_spend["spend"], 1)[0]
                )  # Convert to float
            else:
                slope = 0.0  # No trend if not enough data

            # Calculate purchase count (total number of purchase weeks)
            purchase_count = int(unique_weeks)  # Convert to int

            # Calculate engagement score
            # Higher score means less likely to churn (lower risk)
            # Normalize components to 0-1 range for weighted scoring

            # Recency: lower is better (more recent)
            # Frequency: lower is better (more frequent)
            # Monetary: higher is better (spends more)
            # Trend: higher is better (increasing spend)
            # Purchase count: higher is better (more purchases)

            # Simple churn score: higher means higher risk of churn
            churn_score = (
                (0.4 * min(recency / recency_threshold, 1))
                + (0.3 * min(frequency / 4, 1))
                - (0.1 * min(monetary / 100, 1))
                - (0.1 * (1 if slope > 0 else 0))
                - (0.1 * min(purchase_count / total_weeks, 1))
            )

            # Scale to 0-100%
            churn_risk = float(max(0, min(100, churn_score * 100)))  # Convert to float

            # Determine status
            if recency > recency_threshold:
                status = "Churned"
            elif churn_risk > 60:
                status = "High Risk"
            elif churn_risk > 30:
                status = "Medium Risk"
            else:
                status = "Low Risk"

            # Store metrics
            household_metrics.append(
                {
                    "household_id": int(hshd),  # Convert to int
                    "recency": int(recency),  # Convert to int
                    "frequency": round(float(frequency), 2),  # Convert to float
                    "monetary": round(float(monetary), 2),  # Convert to float
                    "trend": round(float(slope), 2),  # Convert to float
                    "purchase_count": int(purchase_count),  # Convert to int
                    "churn_risk": round(float(churn_risk), 1),  # Convert to float
                    "status": status,
                }
            )

        # 3. Sort households by churn risk (highest first)
        household_metrics.sort(key=lambda x: x["churn_risk"], reverse=True)

        # 4. Get household demographic information for at-risk households
        high_risk_households = [
            h["household_id"]
            for h in household_metrics
            if h["status"] in ["High Risk", "Churned"]
        ]
        households_info = []

        if high_risk_households:
            for hshd_num in high_risk_households[:20]:  # Get top 20 at-risk households
                household = Household.query.filter_by(hshd_num=hshd_num).first()
                if household:
                    households_info.append(
                        {
                            "household_id": household.hshd_num,
                            "loyalty_flag": household.loyalty_flag,
                            "age_range": household.age_range,
                            "income_range": household.income_range,
                            "hshd_composition": household.hshd_composition,
                        }
                    )

        # 5. Analyze demographic patterns in at-risk customers
        demographic_counts = {
            "loyalty": {"Y": 0, "N": 0},
            "age_range": {},
            "income_range": {},
            "composition": {},
        }

        for h in households_info:
            # Count loyalty flag
            if h["loyalty_flag"]:
                demographic_counts["loyalty"][h["loyalty_flag"]] = (
                    demographic_counts["loyalty"].get(h["loyalty_flag"], 0) + 1
                )

            # Count age ranges
            if h["age_range"]:
                demographic_counts["age_range"][h["age_range"]] = (
                    demographic_counts["age_range"].get(h["age_range"], 0) + 1
                )

            # Count income ranges
            if h["income_range"]:
                demographic_counts["income_range"][h["income_range"]] = (
                    demographic_counts["income_range"].get(h["income_range"], 0) + 1
                )

            # Count household compositions
            if h["hshd_composition"]:
                demographic_counts["composition"][h["hshd_composition"]] = (
                    demographic_counts["composition"].get(h["hshd_composition"], 0) + 1
                )

        # print(f"Demographic counts: {demographic_counts}")
        # 6. Calculate risk distribution
        risk_distribution = {
            "Low Risk": len(
                [h for h in household_metrics if h["status"] == "Low Risk"]
            ),
            "Medium Risk": len(
                [h for h in household_metrics if h["status"] == "Medium Risk"]
            ),
            "High Risk": len(
                [h for h in household_metrics if h["status"] == "High Risk"]
            ),
            "Churned": len([h for h in household_metrics if h["status"] == "Churned"]),
        }

        # 7. Prepare chart data
        recency_vs_monetary = [
            {"x": h["recency"], "y": h["monetary"], "risk": h["status"]}
            for h in household_metrics
        ]

        # Format data for charts
        result = {
            "riskDistribution": {
                "labels": list(risk_distribution.keys()),
                "data": list(risk_distribution.values()),
            },
            "highRiskHouseholds": household_metrics[:20],  # Top 20 at-risk households
            "demographicPatterns": demographic_counts,
            "recencyVsMonetary": recency_vs_monetary,
        }

        return jsonify(result)
    except Exception as e:
        print(f"Error in churn prediction: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    with app.app_context():
        print("Creating database tables...")
        db.create_all()

        # Check if data needs to be loaded
        if Household.query.count() == 0:
            print("No data found in database. Loading data...")
            load_data()
        else:
            print("Data already exists in database:")
            print(f"Households: {Household.query.count()}")
            print(f"Transactions: {Transaction.query.count()}")
            print(f"Products: {Product.query.count()}")

            # If we have households but no transactions or products, reload data
            if Household.query.count() > 0 and (
                Transaction.query.count() == 0 or Product.query.count() == 0
            ):
                print("Incomplete data found. Reloading data...")
                # Clear existing data
                Transaction.query.delete()
                Product.query.delete()
                db.session.commit()
                load_data()

    print("Starting Flask application...")
    app.run(debug=True)
