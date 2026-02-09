# AWS PostgreSQL RDS Pulumi Infrastructure

A comprehensive Pulumi template for provisioning a production-ready PostgreSQL database on AWS RDS using TypeScript. This template includes VPC networking, security groups, encryption, automated backups, monitoring, and a bastion host for secure database access and SQL backup restoration.

## Prerequisites

- Pulumi CLI (>= v3): https://www.pulumi.com/docs/get-started/install/
- Node.js (>= 14): https://nodejs.org/
- AWS credentials configured (e.g., via `aws configure` or environment variables)
- AWS EC2 Key Pair created for bastion host access

## Features

- **VPC Infrastructure**: Custom VPC with public and private subnets across multiple availability zones
- **RDS PostgreSQL 15**: Fully configured database instance with encryption at rest
- **Security**: KMS encryption, security groups, and network isolation
- **High Availability**: Optional Multi-AZ deployment support
- **Automated Backups**: Configurable retention period with point-in-time recovery
- **Monitoring**: CloudWatch alarms for CPU and storage, Performance Insights enabled
- **Bastion Host**: EC2 instance for secure database access and SQL restoration
- **Secrets Management**: Secure password generation and output handling

## Getting Started

### 1. Clone the Project

```bash
git clone https://github.com/raghavendramallela/pulumi-aws-psql-sample.git
cd pulumi-aws-psql-sample
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Create EC2 Key Pair

Before deploying, create an EC2 key pair in your AWS region for bastion host access:

```bash
aws ec2 create-key-pair --key-name psql-bastion-key --query 'KeyMaterial' --output text > psql-bastion-key.pem
chmod 400 psql-bastion-key.pem
```

### 4. Configure Your Stack (optional)

Edit `Pulumi.dev.yaml` or use the CLI to set configuration values:

```bash
pulumi config set aws:region us-east-1
pulumi config set dbName myappdb
pulumi config set dbUsername dbadmin
pulumi config set instanceClass db.t3.micro
pulumi config set keyPairName psql-bastion-key
pulumi config set sqlBackupPath ./backup.sql
```

### 5. Deploy Infrastructure

```bash
# Preview changes
pulumi preview

# Deploy the stack
pulumi up
```

### 6. Retrieve Connection Details

```bash
# Get all outputs
pulumi stack output

# Get database endpoint
pulumi stack output dbInstanceEndpoint

# Get database password (sensitive)
pulumi stack output dbPassword_output --show-secrets

# Get full connection string
pulumi stack output connectionString --show-secrets

# Get bastion host IP
pulumi stack output bastionPublicIp
```

## Project Layout

- `Pulumi.yaml` — Pulumi project metadata
- `Pulumi.dev.yaml` — Stack-specific configuration
- `index.ts` — Main infrastructure definition
- `package.json` — Node.js dependencies
- `tsconfig.json` — TypeScript compiler options
- `backup.sql` — Your PostgreSQL backup file (optional)

## Configuration

| Key                      | Description                                    | Default       |
| ------------------------ | ---------------------------------------------- | ------------- |
| `aws:region`             | AWS region to deploy resources                 | `us-east-1`   |
| `dbName`                 | Database name                                  | `myappdb`     |
| `dbUsername`             | Master database username                       | `dbadmin`     |
| `dbPort`                 | PostgreSQL port                                | `5432`        |
| `allocatedStorage`       | Initial storage size in GB                     | `20`          |
| `maxAllocatedStorage`    | Maximum storage for autoscaling in GB          | `100`         |
| `instanceClass`          | RDS instance type                              | `db.t3.micro` |
| `backupRetentionPeriod`  | Number of days to retain backups               | `7`           |
| `multiAz`                | Enable Multi-AZ deployment                     | `false`       |
| `publiclyAccessible`     | Make database publicly accessible              | `false`       |
| `keyPairName`            | EC2 key pair name for bastion host             | (required)    |
| `sqlBackupPath`          | Path to SQL backup file for restoration        | (optional)    |

Use `pulumi config set <key> <value>` to customize configuration.

## Restoring SQL Backup

### Method 1: Manual Restore via Bastion

```bash
# Get connection details
BASTION_IP=$(pulumi stack output bastionPublicIp)
DB_ENDPOINT=$(pulumi stack output dbInstanceAddress)
DB_PASSWORD=$(pulumi stack output dbPassword_output --show-secrets)

# Copy SQL file to bastion
scp -i psql-bastion-key.pem backup.sql ec2-user@$BASTION_IP:~/

# SSH to bastion and restore
ssh -i psql-bastion-key.pem ec2-user@$BASTION_IP

# On bastion host
PGPASSWORD='<your-db-password>' psql -h <db-endpoint> -U dbadmin -d myappdb -f ~/backup.sql
```

### Method 2: Automated Restore Script

Create `restore-db.sh`:

```bash
#!/bin/bash
BASTION_IP=$(pulumi stack output bastionPublicIp)
DB_ENDPOINT=$(pulumi stack output dbInstanceAddress)
DB_PASSWORD=$(pulumi stack output dbPassword_output --show-secrets)
SQL_FILE="backup.sql"

echo "Copying SQL file to bastion..."
scp -i psql-bastion-key.pem $SQL_FILE ec2-user@$BASTION_IP:~/

echo "Restoring database..."
ssh -i psql-bastion-key.pem ec2-user@$BASTION_IP \
  "PGPASSWORD='$DB_PASSWORD' psql -h $DB_ENDPOINT -U dbadmin -d myappdb -f ~/$SQL_FILE"

echo "Restore complete!"
```

Make it executable and run:

```bash
chmod +x restore-db.sh
./restore-db.sh
```

## Infrastructure Components

### Networking
- VPC with CIDR 10.0.0.0/16
- 2 public subnets (10.0.1.0/24, 10.0.2.0/24)
- 2 private subnets (10.0.11.0/24, 10.0.12.0/24)
- Internet Gateway and route tables
- DB subnet group for RDS

### Database
- PostgreSQL 15.4 on RDS
- KMS encryption at rest
- Automated backups with configurable retention
- Performance Insights enabled
- CloudWatch Logs export (postgresql, upgrade)
- Custom parameter group with logging enabled
- Storage autoscaling (gp3)

### Security
- Security groups with minimal required access
- KMS key with automatic rotation
- Secrets stored securely in Pulumi state
- Network isolation in private subnets
- Bastion host for controlled access

### Monitoring
- CloudWatch alarm for CPU utilization (>80%)
- CloudWatch alarm for low storage (<5GB)
- Performance Insights with 7-day retention
- Database connection and query logging

## Managing Your Stack

### Update Configuration

```bash
# Change instance class
pulumi config set instanceClass db.t3.small

# Enable Multi-AZ
pulumi config set multiAz true

# Update and apply changes
pulumi up
```

### View Stack State

```bash
# List all stacks
pulumi stack ls

# View current stack outputs
pulumi stack output

# Export stack state
pulumi stack export > stack-backup.json
```

### Destroy Infrastructure

```bash
# Preview destruction
pulumi destroy --preview

# Destroy all resources
pulumi destroy

# Remove stack completely
pulumi stack rm dev
```

## Security Best Practices

1. **Restrict Bastion Access**: Update bastion security group to allow SSH only from your IP
2. **Rotate Credentials**: Regularly rotate database passwords
3. **Enable Deletion Protection**: Set to `true` in production (already enabled)
4. **Review Security Groups**: Ensure minimal required access
5. **Monitor Logs**: Regularly review CloudWatch logs for suspicious activity
6. **Backup Verification**: Test restore procedures periodically

## Troubleshooting

### Cannot Connect to Database

- Verify security group rules allow traffic from bastion
- Check VPC routing and subnet configuration
- Ensure bastion host has PostgreSQL client installed
- Verify database is in "available" state

### Bastion Host SSH Issues

- Confirm key pair name matches configuration
- Verify key file permissions (`chmod 400`)
- Check security group allows SSH from your IP

### Backup Restore Fails

- Verify SQL file format is compatible with PostgreSQL 15
- Check database user has necessary permissions
- Ensure sufficient storage space available
- Review PostgreSQL logs in CloudWatch

## Cost Optimization

- Use `db.t3.micro` or `db.t4g.micro` for development
- Reduce `backupRetentionPeriod` for non-production environments
- Disable `multiAz` for development stacks
- Set appropriate `maxAllocatedStorage` limits
- Consider Reserved Instances for production workloads

## Next Steps

- Configure automated snapshots schedule
- Set up CloudWatch dashboard for monitoring
- Implement database connection pooling (e.g., RDS Proxy)
- Add read replicas for scaling read operations
- Integrate with AWS Secrets Manager for credential rotation
- Set up VPC peering for application connectivity
- Configure AWS Backup for centralized backup management

## Documentation Help

- Pulumi Documentation: https://www.pulumi.com/docs/
- AWS RDS Documentation: https://docs.aws.amazon.com/rds/
- PostgreSQL Documentation: https://www.postgresql.org/docs/

If you encounter issues, check CloudWatch logs or open an issue in this repository.

