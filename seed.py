from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Hall, UserType, HallUser

def seed_database():
    with SessionLocal() as db:
        try:
            # create user types
            bachelor = UserType(type_name="Bachelor", description="Undergraduate student")
            postgraduate = UserType(type_name="Postgraduate", description="Postgraduate student")
            db.add_all([bachelor, postgraduate])
            db.flush() # to get IDs before committing

            # create halls
            hall1 = Hall(name="Maple Hall", location="North Campus", capacity=200)
            hall2 = Hall(name="Oak Hall", location="South Campus", capacity=150)
            db.add_all([hall1, hall2])
            db.flush() # to get IDs before committing

            # create users
            user1 = HallUser(
                username="john_doe",
                email="johndoe@example.com",
                full_name="John Doe",
                hall_id=hall1.id,
                user_type_id=bachelor.id
            )
            user2 = HallUser(
                username="jane_smith",
                email="janesmith@example.com",
                full_name="Jane Smith",
                hall_id=hall2.id,
                user_type_id=postgraduate.id
            )
            db.add_all([user1, user2])
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error seeding database: {e}")
            raise

if __name__ == "__main__":
    seed_database()