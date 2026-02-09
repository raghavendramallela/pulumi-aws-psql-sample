import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";
import * as random from "@pulumi/random";

// Configuration
const config = new pulumi.Config();
const dbName = config.get("dbName") || "myappdb";
const dbUsername = config.get("dbUsername") || "dbadmin";
const dbPort = config.getNumber("dbPort") || 5432;
const allocatedStorage = config.getNumber("allocatedStorage") || 20;
const maxAllocatedStorage = config.getNumber("maxAllocatedStorage") || 100;
const instanceClass = config.get("instanceClass") || "db.t3.micro";
const backupRetentionPeriod = config.getNumber("backupRetentionPeriod") || 7;
const multiAz = config.getBoolean("multiAz") || false;
const publiclyAccessible = config.getBoolean("publiclyAccessible") || false;
const sqlBackupPath = config.get("sqlBackupPath"); // Path to your SQL backup file

// Generate a random password for the database
const dbPassword = new random.RandomPassword("db-password", {
    length: 32,
    special: true,
    overrideSpecial: "!#$%&*()-_=+[]{}<>:?",
});

// Create VPC for the database
const vpc = new aws.ec2.Vpc("psql-vpc", {
    cidrBlock: "10.0.0.0/16",
    enableDnsHostnames: true,
    enableDnsSupport: true,
    tags: {
        Name: "psql-vpc",
    },
});

// Create Internet Gateway
const igw = new aws.ec2.InternetGateway("psql-igw", {
    vpcId: vpc.id,
    tags: {
        Name: "psql-igw",
    },
});

// Create subnets in different availability zones
const availabilityZones = aws.getAvailabilityZones({
    state: "available",
});

const publicSubnet1 = new aws.ec2.Subnet("psql-public-subnet-1", {
    vpcId: vpc.id,
    cidrBlock: "10.0.1.0/24",
    availabilityZone: availabilityZones.then(azs => azs.names[0]),
    mapPublicIpOnLaunch: true,
    tags: {
        Name: "psql-public-subnet-1",
    },
});

const publicSubnet2 = new aws.ec2.Subnet("psql-public-subnet-2", {
    vpcId: vpc.id,
    cidrBlock: "10.0.2.0/24",
    availabilityZone: availabilityZones.then(azs => azs.names[1]),
    mapPublicIpOnLaunch: true,
    tags: {
        Name: "psql-public-subnet-2",
    },
});

const privateSubnet1 = new aws.ec2.Subnet("psql-private-subnet-1", {
    vpcId: vpc.id,
    cidrBlock: "10.0.11.0/24",
    availabilityZone: availabilityZones.then(azs => azs.names[0]),
    tags: {
        Name: "psql-private-subnet-1",
    },
});

const privateSubnet2 = new aws.ec2.Subnet("psql-private-subnet-2", {
    vpcId: vpc.id,
    cidrBlock: "10.0.12.0/24",
    availabilityZone: availabilityZones.then(azs => azs.names[1]),
    tags: {
        Name: "psql-private-subnet-2",
    },
});

// Create route table for public subnets
const publicRouteTable = new aws.ec2.RouteTable("psql-public-rt", {
    vpcId: vpc.id,
    routes: [
        {
            cidrBlock: "0.0.0.0/0",
            gatewayId: igw.id,
        },
    ],
    tags: {
        Name: "psql-public-rt",
    },
});

// Associate route table with public subnets
new aws.ec2.RouteTableAssociation("psql-public-rta-1", {
    subnetId: publicSubnet1.id,
    routeTableId: publicRouteTable.id,
});

new aws.ec2.RouteTableAssociation("psql-public-rta-2", {
    subnetId: publicSubnet2.id,
    routeTableId: publicRouteTable.id,
});

// Create DB subnet group
const dbSubnetGroup = new aws.rds.SubnetGroup("psql-subnet-group", {
    subnetIds: [privateSubnet1.id, privateSubnet2.id],
    tags: {
        Name: "psql-subnet-group",
    },
});

// Create security group for RDS
const dbSecurityGroup = new aws.ec2.SecurityGroup("psql-sg", {
    vpcId: vpc.id,
    description: "Security group for PostgreSQL RDS instance",
    ingress: [
        {
            protocol: "tcp",
            fromPort: dbPort,
            toPort: dbPort,
            cidrBlocks: publiclyAccessible ? ["0.0.0.0/0"] : ["10.0.0.0/16"],
            description: "PostgreSQL access",
        },
    ],
    egress: [
        {
            protocol: "-1",
            fromPort: 0,
            toPort: 0,
            cidrBlocks: ["0.0.0.0/0"],
            description: "Allow all outbound traffic",
        },
    ],
    tags: {
        Name: "psql-sg",
    },
});

// Create parameter group for PostgreSQL
const dbParameterGroup = new aws.rds.ParameterGroup("psql-params", {
    family: "postgres15",
    description: "Custom parameter group for PostgreSQL 15",
    parameters: [
        {
            name: "log_connections",
            value: "1",
        },
        {
            name: "log_disconnections",
            value: "1",
        },
        {
            name: "log_duration",
            value: "1",
        },
        {
            name: "shared_preload_libraries",
            value: "pg_stat_statements",
        },
    ],
    tags: {
        Name: "psql-params",
    },
});

// Create option group
const dbOptionGroup = new aws.rds.OptionGroup("psql-options", {
    engineName: "postgres",
    majorEngineVersion: "15",
    optionGroupDescription: "Option group for PostgreSQL 15",
    tags: {
        Name: "psql-options",
    },
});

// Create KMS key for encryption
const kmsKey = new aws.kms.Key("psql-kms-key", {
    description: "KMS key for RDS PostgreSQL encryption",
    deletionWindowInDays: 10,
    enableKeyRotation: true,
    tags: {
        Name: "psql-kms-key",
    },
});

const kmsAlias = new aws.kms.Alias("psql-kms-alias", {
    name: "alias/psql-rds-key",
    targetKeyId: kmsKey.keyId,
});

// Create RDS PostgreSQL instance
const dbInstance = new aws.rds.Instance("psql-instance", {
    identifier: "psql-db-instance",
    engine: "postgres",
    engineVersion: "15.4",
    instanceClass: instanceClass,
    allocatedStorage: allocatedStorage,
    maxAllocatedStorage: maxAllocatedStorage,
    storageType: "gp3",
    storageEncrypted: true,
    kmsKeyId: kmsKey.arn,
    
    dbName: dbName,
    username: dbUsername,
    password: dbPassword.result,
    port: dbPort,
    
    dbSubnetGroupName: dbSubnetGroup.name,
    vpcSecurityGroupIds: [dbSecurityGroup.id],
    parameterGroupName: dbParameterGroup.name,
    optionGroupName: dbOptionGroup.name,
    
    publiclyAccessible: publiclyAccessible,
    multiAz: multiAz,
    
    backupRetentionPeriod: backupRetentionPeriod,
    backupWindow: "03:00-04:00",
    maintenanceWindow: "mon:04:00-mon:05:00",
    
    enabledCloudwatchLogsExports: ["postgresql", "upgrade"],
    
    autoMinorVersionUpgrade: true,
    deletionProtection: true,
    skipFinalSnapshot: false,
    finalSnapshotIdentifier: pulumi.interpolate`psql-final-snapshot-${Date.now()}`,
    
    performanceInsightsEnabled: true,
    performanceInsightsKmsKeyId: kmsKey.arn,
    performanceInsightsRetentionPeriod: 7,
    
    copyTagsToSnapshot: true,
    
    tags: {
        Name: "psql-instance",
        Environment: pulumi.getStack(),
    },
});

// Create bastion host for database access (optional, for SQL restore)
const bastionSecurityGroup = new aws.ec2.SecurityGroup("bastion-sg", {
    vpcId: vpc.id,
    description: "Security group for bastion host",
    ingress: [
        {
            protocol: "tcp",
            fromPort: 22,
            toPort: 22,
            cidrBlocks: ["0.0.0.0/0"], // Restrict this to your IP in production
            description: "SSH access",
        },
    ],
    egress: [
        {
            protocol: "-1",
            fromPort: 0,
            toPort: 0,
            cidrBlocks: ["0.0.0.0/0"],
        },
    ],
    tags: {
        Name: "bastion-sg",
    },
});

// Allow bastion to access RDS
new aws.ec2.SecurityGroupRule("bastion-to-rds", {
    type: "ingress",
    fromPort: dbPort,
    toPort: dbPort,
    protocol: "tcp",
    sourceSecurityGroupId: bastionSecurityGroup.id,
    securityGroupId: dbSecurityGroup.id,
    description: "Allow bastion to access RDS",
});

// Get latest Amazon Linux 2 AMI
const ami = aws.ec2.getAmi({
    mostRecent: true,
    owners: ["amazon"],
    filters: [
        {
            name: "name",
            values: ["amzn2-ami-hvm-*-x86_64-gp2"],
        },
    ],
});

// Create bastion host
const bastionHost = new aws.ec2.Instance("bastion-host", {
    instanceType: "t3.micro",
    ami: ami.then(ami => ami.id),
    subnetId: publicSubnet1.id,
    vpcSecurityGroupIds: [bastionSecurityGroup.id],
    keyName: config.get("keyPairName"), // You need to create this key pair in AWS
    
    userData: pulumi.interpolate`#!/bin/bash
yum update -y
yum install -y postgresql15
`,
    
    tags: {
        Name: "psql-bastion-host",
    },
});

// Create CloudWatch alarms
const cpuAlarm = new aws.cloudwatch.MetricAlarm("psql-cpu-alarm", {
    comparisonOperator: "GreaterThanThreshold",
    evaluationPeriods: 2,
    metricName: "CPUUtilization",
    namespace: "AWS/RDS",
    period: 300,
    statistic: "Average",
    threshold: 80,
    alarmDescription: "Alert when CPU exceeds 80%",
    dimensions: {
        DBInstanceIdentifier: dbInstance.identifier,
    },
});

const storageAlarm = new aws.cloudwatch.MetricAlarm("psql-storage-alarm", {
    comparisonOperator: "LessThanThreshold",
    evaluationPeriods: 1,
    metricName: "FreeStorageSpace",
    namespace: "AWS/RDS",
    period: 300,
    statistic: "Average",
    threshold: 5000000000, // 5GB in bytes
    alarmDescription: "Alert when free storage is less than 5GB",
    dimensions: {
        DBInstanceIdentifier: dbInstance.identifier,
    },
});

// Export outputs
export const vpcId = vpc.id;
export const dbInstanceEndpoint = dbInstance.endpoint;
export const dbInstanceAddress = dbInstance.address;
export const dbInstancePort = dbInstance.port;
export const dbName_output = dbInstance.dbName;
export const dbUsername_output = dbInstance.username;
export const dbPassword_output = pulumi.secret(dbPassword.result);
export const bastionPublicIp = bastionHost.publicIp;
export const connectionString = pulumi.secret(
    pulumi.interpolate`postgresql://${dbUsername}:${dbPassword.result}@${dbInstance.endpoint}/${dbName}`
);
