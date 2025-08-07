# Authentication App

This app handles basic authentication functionality for the video consultation API.

## Features

- User registration and login
- JWT token authentication
- User logout

## API Endpoints

### Authentication
- `POST /api/auth/register/` - Register a new user
- `POST /api/auth/login/` - Login user and get JWT tokens
- `POST /api/auth/logout/` - Logout user (invalidate token)
- `POST /api/auth/token/` - Get JWT tokens (alternative to login)

### Usage

#### Register a new user
```json
POST /api/auth/register/
{
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "phone_number": "1234567890",
    "password": "securepassword",
    "role": "user",
    "gender": "male"
}
```

#### Login
```json
POST /api/auth/login/
{
    "email": "user@example.com",
    "password": "securepassword"
}
```

#### Using JWT tokens
Include the access token in the Authorization header:
```
Authorization: Bearer <access_token>
```

## Security Notes
- Use HTTPS in production 