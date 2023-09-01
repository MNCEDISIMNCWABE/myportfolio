
def get_birthday_message(name: str, today: str) -> str:
    if people_birthdays.get(name) == today:
        return f":tada: Happy Birthday, {name}! :tada: Sending you best wishes on your special day! :confetti_ball:"
